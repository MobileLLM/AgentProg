from __future__ import annotations
from dataclasses import dataclass
import os
import re
from typing import Any, Callable, Literal, Tuple
from agentprog.all_utils.fm import FoundationModel
import subprocess
from abc import ABC, abstractmethod
from typing import Optional
import time
import structlog
from PIL import Image

from agentprog.all_utils.general_utils import InitResponseArgs, make_user

from agentprog.all_utils import log_utils
logger = log_utils.get_logger(__name__)

def get_date_on_mobile_phone(serial: str):
    res = subprocess.run(
            [
                "adb",
                "-s",
                serial,
                "shell",
                "date"
            ],
            capture_output=True,
            text=True
        )
    return res.stdout

def get_text_description(screenshot, llm: FoundationModel):
    vlm_single_prompt = '''
"Describe what you see on the current screen in as much detail as possible. 
Requirements: 
1. Every element, especially the icons, should be mentioned in your response. 
2. You should describe the function, text, color, and position of every element, following a top-to-bottom and left-to-right order. 
3. Additionally, if there is a popup window, be sure to emphasize its presence and explain whether it blocks the background elements. 
4. You do not need to output the content in the Position Locator (A horizontal status bar in blue with data labels and values like: \'P: 0 / 1\', \'dX: 0.0\', \'dY: 0.0\', \'Xv: 0.0\', \'Yv: 0.0\', \'Prs: 1.0\' in alternating black or red text on multicolored backgrounds (blue and red for \'Prs\'). The information in the Position Locator is meaningless, please ignore it.
5. You should respond with a string split by ";". An example:
"top-left: App title 'Contacts' in bold white text on a dark blue background; To the right of the app title: White search icon; ...". Don't output a json dict, just a string that can be parsed as json as the answer is ok (using double quotes to enclose it).
6. Answer in English.
'''

    description = llm.query(screenshot, vlm_single_prompt, returns=("description of the current screen.", str))
    return description

"""
Mobile API - Unified interface for mobile device operations
Combines view locator and device controller into a single API
"""


_PATTERN_TO_ACTIVITY = {
    'google chrome|chrome': (
        'com.android.chrome/com.google.android.apps.chrome.Main'
    ),
    'google chat': 'com.google.android.apps.dynamite/com.google.android.apps.dynamite.startup.StartUpActivity',
    'settings|system settings': 'com.android.settings/.Settings',
    'youtube|yt': 'com.google.android.youtube/com.google.android.apps.youtube.app.WatchWhileActivity',
    'google play|play store|gps': (
        'com.android.vending/com.google.android.finsky.activities.MainActivity'
    ),
    'gmail|gemail|google mail|google email|google mail client': (
        'com.google.android.gm/.ConversationListActivityGmail'
    ),
    'google maps|gmaps|maps|google map': (
        'com.google.android.apps.maps/com.google.android.maps.MapsActivity'
    ),
    'google photos|gphotos|photos|google photo|google pics|google images': 'com.google.android.apps.photos/com.google.android.apps.photos.home.HomeActivity',
    'google calendar|gcal': (
        'com.google.android.calendar/com.android.calendar.AllInOneActivity'
    ),
    'camera': 'com.android.camera2/com.android.camera.CameraLauncher',
    'audio recorder': 'com.dimowner.audiorecorder/com.dimowner.audiorecorder.app.welcome.WelcomeActivity',
    'google drive|gdrive|drive': (
        'com.google.android.apps.docs/.drive.startup.StartupActivity'
    ),
    'google keep|gkeep|keep': (
        'com.google.android.keep/.activities.BrowseActivity'
    ),
    'grubhub': (
        'com.grubhub.android/com.grubhub.dinerapp.android.splash.SplashActivity'
    ),
    'tripadvisor': 'com.tripadvisor.tripadvisor/com.tripadvisor.android.ui.launcher.LauncherActivity',
    'starbucks': 'com.starbucks.mobilecard/.main.activity.LandingPageActivity',
    'google docs|gdocs|docs': 'com.google.android.apps.docs.editors.docs/com.google.android.apps.docs.editors.homescreen.HomescreenActivity',
    'google sheets|gsheets|sheets': 'com.google.android.apps.docs.editors.sheets/com.google.android.apps.docs.editors.homescreen.HomescreenActivity',
    'google slides|gslides|slides': 'com.google.android.apps.docs.editors.slides/com.google.android.apps.docs.editors.homescreen.HomescreenActivity',
    'clock': 'com.google.android.deskclock/com.android.deskclock.DeskClock',
    'google search|google': 'com.google.android.googlequicksearchbox/com.google.android.googlequicksearchbox.SearchActivity',
    'contacts': 'com.google.android.contacts/com.android.contacts.activities.PeopleActivity',
    'facebook|fb': 'com.facebook.katana/com.facebook.katana.LoginActivity',
    'whatsapp|wa': 'com.whatsapp/com.whatsapp.Main',
    'instagram|ig': (
        'com.instagram.android/com.instagram.mainactivity.MainActivity'
    ),
    'twitter|tweet': 'com.twitter.android/com.twitter.app.main.MainActivity',
    'snapchat|sc': 'com.snapchat.android/com.snap.mushroom.MainActivity',
    'telegram|tg': 'org.telegram.messenger/org.telegram.ui.LaunchActivity',
    'linkedin': (
        'com.linkedin.android/com.linkedin.android.authenticator.LaunchActivity'
    ),
    'spotify|spot': 'com.spotify.music/com.spotify.music.MainActivity',
    'netflix': 'com.netflix.mediaclient/com.netflix.mediaclient.ui.launch.UIWebViewActivity',
    'amazon shopping|amazon|amzn': (
        'com.amazon.mShop.android.shopping/com.amazon.mShop.home.HomeActivity'
    ),
    'tiktok|tt': 'com.zhiliaoapp.musically/com.ss.android.ugc.aweme.splash.SplashActivity',
    'discord': 'com.discord/com.discord.app.AppActivity$Main',
    'reddit': 'com.reddit.frontpage/com.reddit.frontpage.MainActivity',
    'pinterest': 'com.pinterest/com.pinterest.activity.PinterestActivity',
    'android world': 'com.example.androidworld/.MainActivity',
    'files': 'com.google.android.documentsui/com.android.documentsui.files.FilesActivity',
    'markor': 'net.gsantner.markor/net.gsantner.markor.activity.MainActivity',
    'clipper': 'ca.zgrs.clipper/ca.zgrs.clipper.Main',
    'messages': 'com.google.android.apps.messaging/com.google.android.apps.messaging.ui.ConversationListActivity',
    'simple sms messenger|simple sms|sms messenger': 'com.simplemobiletools.smsmessenger/com.simplemobiletools.smsmessenger.activities.MainActivity',
    'dialer|phone': 'com.google.android.dialer/com.google.android.dialer.extensions.GoogleDialtactsActivity',
    'calendar|simple calendar pro|simple calendar': 'com.simplemobiletools.calendar.pro/com.simplemobiletools.calendar.pro.activities.MainActivity',
    'gallery|simple gallery pro|simple gallery': 'com.simplemobiletools.gallery.pro/com.simplemobiletools.gallery.pro.activities.MainActivity',
    'miniwob': 'com.google.androidenv.miniwob/com.google.androidenv.miniwob.app.MainActivity',
    'draw|simple draw pro': 'com.simplemobiletools.draw.pro/com.simplemobiletools.draw.pro.activities.MainActivity',
    'pro expense|pro expense app': (
        'com.arduia.expense/com.arduia.expense.ui.MainActivity'
    ),
    'broccoli|broccoli app|broccoli recipe app|recipe app': (
        'com.flauschcode.broccoli/com.flauschcode.broccoli.MainActivity'
    ),
    'caa|caa test|context aware access': 'com.google.ccc.hosted.contextawareaccess.thirdpartyapp/.ChooserActivity',
    'osmand': 'net.osmand/net.osmand.plus.activities.MapActivity',
    'tasks|tasks app|tasks.org:': (
        'org.tasks/com.todoroo.astrid.activity.MainActivity'
    ),
    'open tracks sports tracker|activity tracker|open tracks|opentracks': (
        'de.dennisguse.opentracks/de.dennisguse.opentracks.TrackListActivity'
    ),
    'joplin|joplin app': 'net.cozic.joplin/.MainActivity',
    'vlc|vlc app|vlc player': 'org.videolan.vlc/.gui.MainActivity',
    'retro music|retro|retro player': (
        'code.name.monkey.retromusic/.activities.MainActivity'
    ),
}


def get_adb_activity(app_name: str):
  """Get a mapping of regex patterns to ADB activities top Android apps."""
  for pattern, activity in _PATTERN_TO_ACTIVITY.items():
    if re.match(pattern.lower(), app_name.lower()):
      return activity
    
def extract_broadcast_data(raw_output: str) -> Optional[str]:
  """Extracts the data from an adb broadcast command output.

  Args:
    raw_output: The adb command output.

  Returns:
    Extracted data as a string, or None if the result is 0.
  """
  if 'Broadcast completed: result=-1, data=' in raw_output:
    return raw_output.split('data=')[1].strip('"\n')
  elif 'Broadcast completed: result=0' in raw_output:
    return None
  else:
    raise ValueError(f'Unexpected broadcast output: {raw_output}')


class MobileAPIBase:
    """
    Base class for mobile device API that combines locator and executor
    Supports different locator implementations while maintaining the same API interface
    """

    # ==================== Android-specific APIs ====================
    
    def start_app(self, app_name: str) -> bool:
        """
        Open the app named app_name (Android only)
        
        Args:
            app_name: Name of the app to start
            
        Returns:
            bool: True if successful
        """
        raise NotImplementedError()

    def kill_app(self, app_name: str) -> bool:
        """
        Kill the app named app_name (Android only)
        
        Args:
            app_name: Name of the app to kill
            
        Returns:
            bool: True if successful
        """
        raise NotImplementedError()

        
    def _locate_element(self, view_description: str):
        """
        Internal method to locate an element and return coordinates
        
        Args:
            view_description: Description of the view to locate
            
        Returns:
            tuple: (x, y) coordinates of the element center
        """
        raise NotImplementedError()

    def _long_touch(self, x, y, duration=None) -> bool:
        raise NotImplementedError()


    def _click(self, x, y):
        raise NotImplementedError()


    def _clear(self):
        raise NotImplementedError()

    def _get_width_height(self):
        raise NotImplementedError()

    def _do_drag(self, start_xy, end_xy, duration=None) -> bool:
        raise NotImplementedError()

    def _clear_and_input(self, text):
        raise NotImplementedError()

    def _view_set_text(self, text) -> bool:
        raise NotImplementedError()

    def _view_append_text(self, text: str) -> bool:
        raise NotImplementedError()


    def _view_paste_text(self, text: str) -> bool:
        raise NotImplementedError()

    def get_input_field_text(self, view_description: str) -> str:
        """
        Get the text from the input field specified by view_description
        
        Args:
            view_description: Description of the input field
            
        Returns:
            str: Text content of the input field
        """
        raise NotImplementedError()

    def get_clipboard(self) -> str:
        raise NotImplementedError()

    def set_clipboard(self, text: str) -> bool:
        raise NotImplementedError()

    def expand_notification_panel(self):
        raise NotImplementedError()

    def take_screenshot(self, save_path=None) -> Image.Image:
        raise NotImplementedError()

    def back(self):
        raise NotImplementedError()

    def home(self):
        raise NotImplementedError()

    def back_to(self, description: str, max_steps: int = 5) -> bool:
        """
        Navigate back to the view described by description
        
        Args:
            description: Description of the target view
            max_steps: Maximum number of back steps to try
            
        Returns:
            bool: True if reached target, False otherwise
        """
        raise NotImplementedError()

    def swipe_upward(self, view_description: str, distance: Optional[int] = None) -> bool:
        """
        Swipe up on the view specified by view_description
        
        Args:
            view_description: Description of the view to swipe
            distance: Distance to swipe (in pixels), default is screen height / 3
            
        Returns:
            bool: True if successful
        """
        raise NotImplementedError()

    def swipe_downward(self, view_description: str, distance: Optional[int] = None) -> bool:
        """
        Swipe down on the view specified by view_description
        
        Args:
            view_description: Description of the view to swipe
            distance: Distance to swipe (in pixels), default is screen height / 3
            
        Returns:
            bool: True if successful
        """
        raise NotImplementedError()

    def swipe_leftward(self, view_description: str, distance: Optional[int] = None) -> bool:
        """
        Swipe left on the view specified by view_description
        
        Args:
            view_description: Description of the view to swipe
            distance: Distance to swipe (in pixels), default is screen width / 3
            
        Returns:
            bool: True if successful
        """
        raise NotImplementedError()

    def swipe_rightward(self, view_description: str, distance: Optional[int] = None) -> bool:
        """
        Swipe right on the view specified by view_description
        
        Args:
            view_description: Description of the view to swipe
            distance: Distance to swipe (in pixels), default is screen width / 3
            
        Returns:
            bool: True if successful
        """
        raise NotImplementedError()

    def swipe_until(
        self,
        view_description: str,
        expected_desc: str,
        towards: str = "up",
        duration: int = 1000,
        max_retry: int = 10
    ) -> bool:
        """
        Swipe the view specified by view_description until expected_desc is fulfilled
        
        Args:
            view_description: Description of the view to swipe
            expected_desc: Description of the expected result
            towards: Direction to swipe ("up", "down", "left", "right")
            duration: Duration of each swipe in milliseconds
            max_retry: Maximum number of swipes to try
            
        Returns:
            bool: True if expected view appears, False otherwise
        """
        raise NotImplementedError()

    def wait_until(
        self,
        description: str,
        waitInterval: float = 0.5,
        timeout: float = 5
    ) -> bool:
        """
        Wait for a view described by description to appear
        
        Args:
            description: Description of the view to wait for
            waitInterval: Interval between checks in seconds
            timeout: Maximum time to wait in seconds (-1 means unlimited)
            
        Returns:
            bool: True if view appears, False otherwise
        """
        raise NotImplementedError()

    def check(self, description: str) -> bool:
        """
        Check whether the current screen state matches description
        
        Args:
            description: Description of the expected screen state
            
        Returns:
            bool: True if matched, False otherwise
        """
        raise NotImplementedError()

    def long_click(self, view_description: str) -> bool:
        """
        Long click the view specified by view_description for 1 second
        
        Args:
            view_description: Description of the view to long click
            
        Returns:
            bool: True if successful
        """
        raise NotImplementedError()

    
    def click(self, view_description: str) -> bool:
        """
        Click the view specified by view_description
        
        Args:
            view_description: Description of the view to click
            
        Returns:
            bool: True if successful
        """
        raise NotImplementedError()

    def input(self, view_description: str, text: str) -> bool:
        """
        Clear the input field specified by view_description and input the given text
        You don't have to call a keyboard; use this input method directly
        
        Args:
            view_description: Description of the input field
            text: Text to input
            
        Returns:
            bool: True if successful
        """
        raise NotImplementedError()

    def input_by_pasting(self, view_description: str, text: str) -> bool:
        """
        Input text into the view specified by view_description by pasting
        Use this when standard input doesn't work, such as in the WeChat app
        
        Args:
            view_description: Description of the input field
            text: Text to input by pasting
            
        Returns:
            bool: True if successful
        """
        raise NotImplementedError()

@dataclass
class MobileAPIConfig:
    locator: Literal["ui_tars", "aw"]
    device_serial_id: str
    llm: FoundationModel = None
    locator_config: Any = None
    tensorboard_log_dir: str = None

class LocatorAPI(ABC):
    @abstractmethod
    def locate(self, view_description):
        pass

    @abstractmethod
    def check(self, view_description):
        pass

class UiTarsLocator(LocatorAPI):
    def __init__(self, config: MobileAPIConfig, mobile_api: MobileAPI):
        from agentprog.all_utils.ui_tars_utils import init_get_ui_tars_response
        self.config = config
        self.mobile_api = mobile_api
        self.IMAGE_FACTOR = 28
        self.MIN_PIXELS = 100 * 28 * 28
        self.MAX_PIXELS = 16384 * 28 * 28
        self.MAX_RATIO = 200
        
        self.local_mode = False
        self.image_resize_factor = 0.5
        self.language = 'Chinese'
        self.serialize_mode = 'qwen' if self.local_mode else 'openai'
        if self.local_mode:
            get_ui_tars_response = init_get_ui_tars_response(base_url='http://127.0.0.1:8888', api_key='')
        else:
            get_ui_tars_response = init_get_ui_tars_response(init_response_args=InitResponseArgs(model="doubao-1.5-ui-tars-250428", record_completion_statistics=True, tensorboard_log_dir=config.tensorboard_log_dir))
        self.get_ui_tars_response = get_ui_tars_response

    @property
    def width(self):
        return self.mobile_api._get_width_height()[0]
    
    @property
    def width(self):
        return self.mobile_api._get_width_height()[1]
    

    def locate(self, description):
        """
        locate the view using a Vision-Language Model
        """
        from agentprog.all_utils.ui_tars_utils import init_get_ui_tars_response, get_ui_tars_mobile_prompt_local, get_ui_tars_mobile_prompt_api, parse_ui_tars_response, get_ui_tars_messages
        from agentprog.all_utils.general_utils import init_get_parsed_response, init_get_gemini_response
        get_fix_response = init_get_gemini_response(model='gemini-2.5-pro')


        get_ui_tars_mobile_prompt = get_ui_tars_mobile_prompt_local if self.local_mode else get_ui_tars_mobile_prompt_api

        current_screenshot = self.mobile_api.take_screenshot()
        resized_current_screenshot = current_screenshot.resize((round(current_screenshot.width * self.image_resize_factor), round(current_screenshot.height * self.image_resize_factor)))

        get_parsed_response = init_get_parsed_response(self.get_ui_tars_response, lambda r: parse_ui_tars_response(self.local_mode, r, resized_current_screenshot.width, resized_current_screenshot.height, self.image_resize_factor), try_times=3, get_fix_response=get_fix_response)

        messages = get_ui_tars_messages(
            get_ui_tars_mobile_prompt=get_ui_tars_mobile_prompt, 
            language=self.language, 
            instruction=f'Click on {description}.', 
            screenshot=resized_current_screenshot,
            history_list=[],
            serialize_mode=self.serialize_mode
        )
        parsed_dict, response = get_parsed_response(messages)
        logger.debug(f"UI TARS model locate view response: {response}")
        x, y = int(parsed_dict.get("args")['start_box'][0]), int(parsed_dict.get("args")['start_box'][1])
        current_screenshot.close()
        return (x, y, x, y)

    def _get_check_prompt(matching_elements, excluding_elements=None):
        return f"You are a GUI agent. You are given a query and an answer format, with screenshots. Your task is to analyze the screenshots and provide the information requested in the query, strictly following the given answer format.\nQuery: Describe matching elements: {matching_elements} excluding: {excluding_elements}\nAnswer Format: list[str]\nScreenshots: "

    def check(self, description: str | list[str]):
        screenshot = self.mobile_api.take_screenshot()

        if isinstance(description, list):
            condition_list = "[" + ",".join(description) + "]"
        elif isinstance(description, str):
            condition_list = f"[{description}]"
        else:
            raise TypeError(f"Invalid description type: {type(description)}")
        is_success = False
        for _ in range(3):
            try:
                response = self.get_ui_tars_response([make_user(screenshot, self._get_check_prompt([condition_list])).serialize()])
                response = str(response).strip("[] ")
                check_result = re.split(r'\s*,\s*', response)
                is_success = True
                break
            except Exception as _:
                logger.exception(f"Failed to check for \"{condition_list}\" on the current screen, retrying...", action='check', status='retry')
                # 重新截图
                screenshot = self.mobile_api.take_screenshot()
                continue
        if not is_success:
            raise RuntimeError(f"Failed to check for \"{condition_list}\" on the current screen")

        logger.debug(f"Check result: {check_result}")
        check_result = 'true' in str(check_result).lower()
        
        logger.info(f"🔎 Checking for \"{description}\", result is: {check_result}")

        return check_result

def get_locator(config: MobileAPIConfig, mobile_api: MobileAPI):
    match config.locator:
        case 'ui_tars':
            return UiTarsLocator(config, mobile_api)
        case 'aw':
            from agentprog.all_utils.aw_utils import AndroidWorldLocator
            return AndroidWorldLocator(config, mobile_api)

class MobileAPI(MobileAPIBase):
    """
    Base class for mobile device API that combines locator and executor
    Supports different locator implementations while maintaining the same API interface
    """

    def __init__(self, config: MobileAPIConfig):
        """
        Initialize mobile API with agent and device
        
        Args:
        """
        self.config = config
        self.locator: LocatorAPI = get_locator(config, self)  # To be initialized by subclasses
        self.device_serial_id = self.config.device_serial_id
        self.width, self.height = self._get_width_height()

    # ==================== Android-specific APIs ====================
    
    def start_app(self, app_name: str) -> bool:
        """
        Open the app named app_name (Android only)
        
        Args:
            app_name: Name of the app to start
            
        Returns:
            bool: True if successful
        """
        try:
            app_activity_name = get_adb_activity(app_name)
            if app_activity_name is None:
                raise ValueError(f"App Name `{app_name}` not exist!")
            app_package_name = app_activity_name.split("/")[0]
            if app_package_name:
                os.system(f"adb -s {self.device_serial_id} shell am force-stop {app_package_name}")
                os.system(f"adb -s {self.device_serial_id} shell monkey -p {app_package_name} -c android.intent.category.LAUNCHER 1")
                time.sleep(1)
                return True
            else:
                raise ValueError(f"Cannot Find App Name: {app_name}")
        except Exception as e:
            raise e

    def kill_app(self, app_name: str) -> bool:
        """
        Kill the app named app_name (Android only)
        
        Args:
            app_name: Name of the app to kill
            
        Returns:
            bool: True if successful
        """
        try:
            app_activity_name = get_adb_activity(app_name)
            app_package_name = app_activity_name.split("/")[0]
            if app_package_name:
                os.system(f"adb -s {self.device_serial_id} shell am force-stop {app_package_name}")
                return True
            else:
                raise ValueError(f"Cannot Find App Name: {app_name}")
        except Exception as e:
            raise e
        
    def _locate_element(self, view_description: str):
        """
        Internal method to locate an element and return coordinates
        
        Args:
            view_description: Description of the view to locate
            
        Returns:
            tuple: (x, y) coordinates of the element center
        """
        try:
            bound = self.locator.locate(view_description)
            
            if bound == (None, None, None, None):
                raise RuntimeError(f"Could not locate element: {view_description}")

            
            # Calculate center point
            x = (bound[0] + bound[2]) / 2
            y = (bound[1] + bound[3]) / 2
            
            return int(x), int(y)
        except Exception as e:
            logger.error(f"Failed to locate element: {view_description}", error=str(e))

            raise

    def _long_touch(self, x, y, duration=None) -> bool:
        try:
            duration = duration if duration is not None else 400
            os.system(f'adb -s {self.device_serial_id} shell input swipe {x} {y} {x} {y} {duration}')
            time.sleep(1)
            return (x, y)
        except Exception as e:
            return (x, y)

    def _click(self, x, y):
        return self._long_touch(x, y)

    def _clear(self):
        os.system(f"adb -s {self.device_serial_id} shell input keycombination 113 29 && adb -s {self.device_serial_id} shell input keyevent 67")
        return True

    def _get_width_height(self):
        return 1080, 2400

    def _do_drag(self, start_xy, end_xy, duration=None) -> bool:
        os.system(f"adb -s {self.device_serial_id} shell input swipe {start_xy[0]} {start_xy[1]} {end_xy[0]} {end_xy[1]} {duration}")
        self.agent.sleep(duration * 0.001)
        return True

    def _clear_and_input(self, text):
        logger.debug(
            f'Clear and input action done with text: {text}',
            action='clear_and_input',
            status='done',
            metadata={'text': text}
        )
        self._view_set_text(text)

    def _view_set_text(self, text) -> bool:
        self._clear()
        return self._view_append_text(text)

    def _view_append_text(self, text: str) -> bool:
        # res = self._send_command('input,' + text)
        # return True
        if not text.isascii(): # 若非 ascii，重定向至粘贴 text
            return self._view_paste_text(text)
        try:
            # os.system(f'adb shell input text "{text}"')
            # 检查输入文本是否以换行符结尾
            has_trailing_newline = text.endswith("\n")

            # 将字符串按照换行符拆分
            lines = text.splitlines()

            # 遍历所有行并输入文本
            for i, line in enumerate(lines):
                # 将空格替换为ADB要求的转义字符 %s
                escaped_line = line.replace(" ", "%s")
                escaped_line = escaped_line.replace("'", "\\'")
                os.system(f'adb -s {self.device_serial_id} shell input text "{escaped_line}"')
                # 如果不是最后一行，则模拟回车（KEYCODE_ENTER对应66）
                if i < len(lines) - 1:
                    os.system(f'adb -s {self.device_serial_id} shell input keyevent 66')

            # 如果输入的文本最后有换行符，则额外模拟一次回车
            if has_trailing_newline:
                os.system(f'adb -s {self.device_serial_id} shell input keyevent 66')

            return True
        except Exception as e:
            print(f"Error in view_set_text: {str(e)}")
            return False


    def _view_paste_text(self, text: str) -> bool:
        old_text = self.get_clipboard()
        self.set_clipboard(text)
        res = os.system(f"adb -s {self.device_serial_id} shell input keyevent KEYCODE_PASTE")
        self.set_clipboard(old_text)
        if res == os.EX_OK:
            return True
        else:
            return False
    # ==================== Universal APIs ====================

    def get_input_field_text(self, view_description: str) -> str:
        """
        Get the text from the input field specified by view_description
        
        Args:
            view_description: Description of the input field
            
        Returns:
            str: Text content of the input field
        """

        logger.info(f"📝 获取输入框文本: {view_description}")
        # Locate the input field first
        x, y = self._locate_element(view_description)
        # Click to focus
        self._click(x, y)
        time.sleep(0.5)
        
        # Select all text and copy to clipboard
        self._clear()  # This will select all
        old_clipboard = self.get_clipboard()
        
        # Copy the selected text
        # Using keyboard shortcut Ctrl+C (key combination)
        import subprocess
        subprocess.run(
            f"adb -s {self.device_serial_id} shell input keyevent 278",  # KEYCODE_COPY
            shell=True,
            check=True
        )
        time.sleep(0.3)
        
        # Get the copied text
        text = self.get_clipboard()
        
        # Restore old clipboard
        self.set_clipboard(old_clipboard)
        
        return text
    
    def get_clipboard(self) -> str:
        self.start_app("clipper")
        res = subprocess.run(
            [
                "adb",
                "-s",
                self.device_serial_id,
                "shell",
                "am",
                "broadcast",
                "-a",
                "clipper.get"
            ],
            capture_output=True,
            text=True
        )
        content = extract_broadcast_data(res.stdout) or ""
        self.back()
        time.sleep(1)
        return content

    def set_clipboard(self, text: str) -> bool:
        self.start_app("clipper")
        text_cleaned = text.replace(" ", "\\ ")
        os.system(f"adb -s {self.device_serial_id} shell am broadcast -a clipper.set -e text \"{text_cleaned}\"")
        self.back()
        time.sleep(1)
        return True
    
    def expand_notification_panel(self):
        os.system(f"adb -s {self.device_serial_id} shell cmd statusbar expand-notifications")
        return True

    def take_screenshot(self, save_path=None) -> Image.Image:
        from pathlib import Path
        for _ in range(3):
            screenshot_path = f"screenshot_{self.device_serial_id}.png"
            result = subprocess.run(
                f"adb -s {self.device_serial_id} exec-out screencap -p > {screenshot_path}",
                shell=True,
                check=True
            )
            screenshot_image = Image.open(screenshot_path)
            try:
                save_path = f"test_screenshot_{self.device_serial_id}.png"
                screenshot_image.save(save_path) # ensure the png is ok
                Path(save_path).unlink()
                return screenshot_image
            except Exception as e:
                continue
        raise e

    def back(self):
        os.system(f"adb -s {self.device_serial_id} shell input keyevent 4")
        return True

    def home(self):
        os.system(f"adb -s {self.device_serial_id} shell input keyevent 3")
        return True

    def back_to(self, description: str, max_steps: int = 5) -> bool:
        """
        Navigate back to the view described by description
        
        Args:
            description: Description of the target view
            max_steps: Maximum number of back steps to try
            
        Returns:
            bool: True if reached target, False otherwise
        """
        logger.info(f"🔙 Navigating back to: {description} (max {max_steps} steps)")

        for step in range(max_steps):
            if self.check(description):
                logger.info(f"✅ Reached the target screen (step {step})")
                return True
            self.back()
            time.sleep(1)
        
        logger.warning(f"⚠️ Could not reach the target screen within {max_steps} steps")
        return False

    def swipe_upward(self, view_description: str, distance: Optional[int] = None) -> bool:
        """
        Swipe up on the view specified by view_description
        
        Args:
            view_description: Description of the view to swipe
            distance: Distance to swipe (in pixels), default is screen height / 3
            
        Returns:
            bool: True if successful
        """
        logger.info(f"⬆️ Swiping upward on: {view_description}")

        x, y = self._locate_element(view_description)
        
        if distance is None:
            distance = self.height // 3
        
        start_x, start_y = x, y
        end_x, end_y = x, y - distance
        
        return self._do_drag((start_x, start_y), (end_x, end_y), duration=300)

    def swipe_downward(self, view_description: str, distance: Optional[int] = None) -> bool:
        """
        Swipe down on the view specified by view_description
        
        Args:
            view_description: Description of the view to swipe
            distance: Distance to swipe (in pixels), default is screen height / 3
            
        Returns:
            bool: True if successful
        """
        logger.info(f"⬇️ Swiping downward on: {view_description}")

        x, y = self._locate_element(view_description)
        
        if distance is None:
            distance = self.height // 3
        
        start_x, start_y = x, y
        end_x, end_y = x, y + distance
        
        return self._do_drag((start_x, start_y), (end_x, end_y), duration=300)

    def swipe_leftward(self, view_description: str, distance: Optional[int] = None) -> bool:
        """
        Swipe left on the view specified by view_description
        
        Args:
            view_description: Description of the view to swipe
            distance: Distance to swipe (in pixels), default is screen width / 3
            
        Returns:
            bool: True if successful
        """
        logger.info(f"⬅️ Swiping leftward on: {view_description}")
        x, y = self._locate_element(view_description)
        
        if distance is None:
            distance = self.width // 3
        
        start_x, start_y = x, y
        end_x, end_y = x - distance, y
        
        return self._do_drag((start_x, start_y), (end_x, end_y), duration=300)

    def swipe_rightward(self, view_description: str, distance: Optional[int] = None) -> bool:
        """
        Swipe right on the view specified by view_description
        
        Args:
            view_description: Description of the view to swipe
            distance: Distance to swipe (in pixels), default is screen width / 3
            
        Returns:
            bool: True if successful
        """
        logger.info(f"➡️ Swiping rightward on: {view_description}")

        x, y = self._locate_element(view_description)
        
        if distance is None:
            distance = self.width // 3
        
        start_x, start_y = x, y
        end_x, end_y = x + distance, y
        
        return self._do_drag((start_x, start_y), (end_x, end_y), duration=300)

    def swipe_until(
        self,
        view_description: str,
        expected_desc: str,
        towards: str = "up",
        duration: int = 1000,
        max_retry: int = 10
    ) -> bool:
        """
        Swipe the view specified by view_description until expected_desc is fulfilled
        
        Args:
            view_description: Description of the view to swipe
            expected_desc: Description of the expected result
            towards: Direction to swipe ("up", "down", "left", "right")
            duration: Duration of each swipe in milliseconds
            max_retry: Maximum number of swipes to try
            
        Returns:
            bool: True if expected view appears, False otherwise
        """
        logger.info(f"🔄 Swiping until element appears: {expected_desc} (direction: {towards}, max {max_retry} retries)")
        
        swipe_methods = {
            "up": self.swipe_upward,
            "down": self.swipe_downward,
            "left": self.swipe_leftward,
            "right": self.swipe_rightward
        }
        
        if towards not in swipe_methods:
            raise ValueError(f"Invalid swipe direction: {towards}")
        
        swipe_method = swipe_methods[towards]
        
        for retry in range(max_retry):
            if self.check(expected_desc):
                logger.info(f"✅ Target element has appeared (on swipe attempt {retry})")
                return True
            
            swipe_method(view_description)
            time.sleep(duration / 1000)
        
        logger.warning(f"⚠️ Did not find the target element after {max_retry} swipes")
        return False

    def wait_until(
        self,
        description: str,
        waitInterval: float = 0.5,
        timeout: float = 5
    ) -> bool:
        """
        Wait for a view described by description to appear
        
        Args:
            description: Description of the view to wait for
            waitInterval: Interval between checks in seconds
            timeout: Maximum time to wait in seconds (-1 means unlimited)
            
        Returns:
            bool: True if view appears, False otherwise
        """
        logger.info(f"⏳ Waiting for element to appear: {description} (timeout: {timeout}s)")
        
        start_time = time.time()
        
        while True:
            if self.check(description):
                logger.info(f"✅ Element has appeared")
                return True
            
            if timeout != -1 and (time.time() - start_time) > timeout:
                logger.warning(f"⚠️ Timed out while waiting")
                return False
            
            time.sleep(waitInterval)

    def check(self, description: str) -> bool:
        """
        Check whether the current screen state matches description
        
        Args:
            description: Description of the expected screen state
            
        Returns:
            bool: True if matched, False otherwise
        """
        logger.debug(f"🔍 检查: {description}")
        return self.locator.check(description)

    def long_click(self, view_description: str) -> bool:
        """
        Long click the view specified by view_description for 1 second
        
        Args:
            view_description: Description of the view to long click
            
        Returns:
            bool: True if successful
        """
        logger.info(f"👆⏱️ Long clicking: {view_description}")
        if "button" not in view_description and "button" not in view_description:
            view_description = f"Clickable component containing {view_description}"

    
    def click(self, view_description: str) -> bool:
        """
        Click the view specified by view_description
        
        Args:
            view_description: Description of the view to click
            
        Returns:
            bool: True if successful
        """
        logger.info(f"👆 Clicking: {view_description}")
        if "button" not in view_description and "button" not in view_description:
            view_description = f"Clickable component containing {view_description}"
        x, y = self._locate_element(view_description)
        return self._click(x, y)

    def input(self, view_description: str, text: str) -> bool:
        """
        Clear the input field specified by view_description and input the given text
        You don't have to call a keyboard; use this input method directly
        
        Args:
            view_description: Description of the input field
            text: Text to input
            
        Returns:
            bool: True if successful
        """
        logger.info(f"⌨️ Inputting text into {view_description}: {text}")
        if "textbox" not in view_description and "input field" not in view_description:
            view_description = f"Text editing component containing {view_description}"
        # Locate and click the input field
        x, y = self._locate_element(view_description)
        self._click(x, y)
        time.sleep(0.5)
        
        # Clear existing text and input new text
        return self._clear_and_input(text)

    def input_by_pasting(self, view_description: str, text: str) -> bool:
        """
        Input text into the view specified by view_description by pasting
        Use this when standard input doesn't work, such as in the WeChat app
        
        Args:
            view_description: Description of the input field
            text: Text to input by pasting
            
        Returns:
            bool: True if successful
        """
        logger.info(f"📋 Inputting text into {view_description} by pasting: {text}")
        
        # Locate and click the input field
        x, y = self._locate_element(view_description)
        self._click(x, y)
        time.sleep(0.5)
        
        # Clear existing text
        self._clear()
        time.sleep(0.3)
        
        # Paste text
        return self._view_paste_text(text)
