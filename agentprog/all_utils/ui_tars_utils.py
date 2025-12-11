import ast
import os
import random
import re
import time

from tenacity import RetryCallState, before_sleep_log, retry
import tenacity
from openai import OpenAI
import openai
from typing import Callable, List, Optional, Tuple
from PIL import Image
from agentprog.all_utils.general_utils import InitResponseArgs, Messages, RichStr, dataclass, init_get_response_with_completion_statistics, make_assistant, make_user, init_get_openai_response, log_retry_error_with_traceback
import subprocess
from agentprog.all_utils.debug import need_breakpoint
from rich.syntax import Syntax
import os
from rich.console import Console
from agentprog.all_utils import log_utils
import logging

# import pyautogui
UI_TARS_MAX_LOOP = 20
ui_tars_client = None # ui tars client
logger = log_utils.get_logger(__name__)

console = Console()
def clear_and_print_pairs(*pairs):
    if not need_breakpoint: return
    # 清空终端
    # os.system("cls" if os.name == "nt" else "clear")
    for before, code in pairs:
        console.print(before)
        # 语法高亮显示 Python 代码
        syntax = Syntax(code, "python", theme="monokai", line_numbers=True, word_wrap=True)
        console.print(syntax)

@dataclass
class UiTarsResult:
    executed_successfully: bool
    finished: bool
    content: str=''
    error_info: Exception = None

@dataclass
class UiTarsLoopResult:
    finished: bool
    content: str = ''
    error_info: str = ''


def convert_point_to_coordinates_four_numbers(text, is_answer=False):
    # 匹配 <point> 后面的四个数字
    pattern = r"<point>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)</point>"
    def replace_match(match):
        x1, y1, x2, y2 = map(int, match.groups())
        x = (x1 + x2) // 2  # 使用截断取整计算中心点x坐标
        y = (y1 + y2) // 2  # 使用截断取整计算中心点y坐标
        if is_answer:
            return f"({x},{y})"  # 只返回 (x, y) 格式
        return f"({x},{y})"  # 返回带标签的格式

    # 去掉 [EOS] 并替换 <point> 坐标
    text = re.sub(r"\[EOS\]", "", text)
    return re.sub(pattern, replace_match, text).strip()

def take_screenshot_adb(serial: str='emulator-5554') -> Image.Image:
    for _ in range(3):
        screenshot_path = f"screenshot_{serial}.png"
        result = subprocess.run(
            f"adb -s {serial} exec-out screencap -p > {screenshot_path}",
            shell=True,
            check=True
        )
        screenshot_image = Image.open(screenshot_path)
        try:
            screenshot_image.save(f"test_screenshot_{serial}.png")
            return screenshot_image
        except Exception as e:
            continue
    raise e

def back(serial):
    os.system(f"adb -s {serial} shell input keyevent 4")  # KEYCODE_BACK
    return True

def paste_text(text: str, serial) -> bool:
    old_text = get_clipboard(serial)
    set_clipboard(text, serial)
    res = os.system(f"adb -s {serial} shell input keyevent KEYCODE_PASTE")
    set_clipboard(old_text, serial)
    if res == os.EX_OK:
        return True
    else:
        return False

def set_clipboard(text, serial) -> bool:
    start_app("clipper", serial=serial)
    text_cleaned = text.replace(" ", "\\ ")
    os.system(f"adb -s {serial} shell am broadcast -a clipper.set -e text \"{text_cleaned}\"")
    back(serial)
    time.sleep(1)
    return True

def get_clipboard(serial: str) -> str:
    def extract_broadcast_data(raw_output: str) -> Optional[str]:
        """Extracts the data from an adb -s {serial} broadcast command output.

        Args:
            raw_output: The adb -s {serial} command output.

        Returns:
            Extracted data as a string, or None if the result is 0.
        """
        if 'Broadcast completed: result=-1, data=' in raw_output:
            return raw_output.split('data=')[1].strip('"\n')
        elif 'Broadcast completed: result=0' in raw_output:
            return None
        else:
            raise ValueError(f'Unexpected broadcast output: {raw_output}')
    # res = self._send_command('get_clipboard')
    start_app("clipper", serial=serial)
    res = subprocess.run(
        [
            "adb",
            "-s",
            serial,
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
    back(serial)
    time.sleep(1)
    return content

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
  """Get a mapping of regex patterns to ADB -s {serial} activities top Android apps."""
  for pattern, activity in _PATTERN_TO_ACTIVITY.items():
    if re.match(pattern.lower(), app_name.lower()):
      return activity

def start_app(app, serial: str) -> bool:
    try:
        app_activity_name = get_adb_activity(app)
        app_package_name = app_activity_name.split("/")[0]
        if app_package_name:
            os.system(f"adb -s {serial} shell monkey -p {app_package_name} -c android.intent.category.LAUNCHER 1")
            time.sleep(2.5)
            return True
        else:
            raise ValueError(f"Cannot Find App Name: {app}")
    except Exception as e:
            raise e

def kill_app(app, serial: str) -> bool:
    try:
        app_activity_name = get_adb_activity(app)
        app_package_name = app_activity_name.split("/")[0]
        if app_package_name:
            os.system(f"adb -s {serial} shell am force-stop {app_package_name}")
            return True
        else:
            raise ValueError(f"Cannot Find App Name: {app}")
    except Exception as e:
        raise e

def init_get_ui_tars_response(base_url: str=None, api_key: str=None, model: str="doubao-1.5-ui-tars-250428", init_response_args: InitResponseArgs=None):
    if init_response_args is not None:
        model = init_response_args.model
        base_url = init_response_args.base_url
        api_key = init_response_args.api_key
    global ui_tars_client
    if ui_tars_client is None:
        logger.info(f"initializing client {OpenAI.__name__}")
        ui_tars_client = OpenAI(
            # 此为默认路径，您可根据业务所在地域进行配置
            base_url=base_url or os.environ.get("DOUBAO_BASE_URL"),
            # 从环境变量中获取您的 API Key。此为默认方式，您可根据需要进行修改
            api_key=api_key or os.environ.get("ARK_API_KEY"),
        )
        logger.info(f"Successfully initialized client {type(ui_tars_client)}")

    logger.info(f"initializing get ui_tars response", ui_tars_client=ui_tars_client, model=model)

    get_ui_tars_response = lambda m: (lambda r: RichStr(r.choices[0].message.content, r))(ui_tars_client.chat.completions.create(
        model=model,    # 通过 endpoint 指定模型
        messages=m,
        temperature=0.2,
        stream=False,
    ))
    get_ui_tars_response = tenacity.retry(stop=tenacity.stop_after_attempt(10), wait=tenacity.wait_fixed(30), before_sleep=before_sleep_log(logger, logging.WARNING))(get_ui_tars_response)
    if init_response_args and init_response_args.record_completion_statistics:
        get_ui_tars_response = init_get_response_with_completion_statistics(get_ui_tars_response, init_response_args)
    return get_ui_tars_response

def get_ui_tars_mobile_prompt_api(language: str, instruction: str):
    '''
    字节火山上提供的 prompt。与 github 的 prompt 不同。二者的 prompt 设计和生成答案的格式也有较大差别。实测即使交换 prompt ，答案的格式也不会改变。
    '''
    return f'''
You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 
## Output Format
```
Thought: ...
Action: ...
```
## Action Space

click(point='<point>x1 y1</point>')
long_press(point='<point>x1 y1</point>')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(point='<point>x1 y1</point>', direction='down or up or right or left')
open_app(app_name=\'\')
drag(start_point='<point>x1 y1</point>', end_point='<point>x2 y2</point>')
press_home()
press_back()
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.


## Note
- Use {language} in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}
'''

def get_ui_tars_mobile_prompt_local(language, instruction):
    '''
    字节 github 仓库的 prompt。与火山的版本有所不同。二者的 prompt 设计和生成答案的格式也有较大差别。实测即使交换 prompt ，答案的格式也不会改变。
    '''
    return f'''
You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 
## Output Format
```
Thought: ...
Action: ...
```
## Action Space

click(point='<|box_start|>(x1, y1)<|box_end|>')
long_press(point='<|box_start|>(x1, y1)<|box_end|>')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(point='<|box_start|>(x1, y1)<|box_end|>', direction='down or up or right or left')
open_app(app_name=\'\')
drag(start_point='<|box_start|>(x1, y1)<|box_end|>', end_point='<|box_start|>(x2, y2)<|box_end|>')
press_home()
press_back()
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.


## Note
- Use {language} in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}
'''

def get_fix_prompt(response: str):
    return f'''
允许调用的 API 为：

click(point='<|box_start|>(x1, y1)<|box_end|>')
long_press(point='<|box_start|>(x1, y1)<|box_end|>')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(point='<|box_start|>(x1, y1)<|box_end|>', direction='down or up or right or left')
open_app(app_name=\'\')
drag(start_point='<|box_start|>(x1, y1)<|box_end|>', end_point='<|box_start|>(x2, y2)<|box_end|>')
press_home()
press_back()
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.

# Your Job
你需要修复当前 API 调用格式中的错误，即 Action 中的格式错误。Thought 不要动，输出修改后的完整内容。
注意：
1. 不要提出任何问题，直接按照要求返回。

需要修复的内容如下：
{response}
'''

def get_ui_tars_messages(get_ui_tars_mobile_prompt, language: str, instruction: str, screenshot: Image.Image, history_list: List[Tuple[Image.Image, str]], max_images=5, serialize_mode='qwen'):
    final_messages = Messages(make_user(get_ui_tars_mobile_prompt(language=language, instruction=instruction)))

    for history_idx, (history_screenshot, history_action) in enumerate(history_list[::-1]):
        final_messages.messages.insert(1, make_assistant(history_action))
        if history_idx < max_images:
            final_messages.messages.insert(1, make_user(history_screenshot))
    final_messages.append(make_user(screenshot))
    
    return final_messages.serialize(mode=serialize_mode)

def parse_ui_tars_response(local_mode: bool, response: str, resized_width, resized_height, resize_factor) -> dict:
    from ui_tars.action_parser import parse_action_to_structure_output, parsing_response_to_pyautogui_code, convert_point_to_coordinates
    response = response.replace("（", "(")
    response = response.replace("）", ")")
    if "<point>" in response: # 统一格式
        response = convert_point_to_coordinates(response)
    if "<point>" in response: # 4 个数字也有可能
        response = convert_point_to_coordinates_four_numbers(response)
    parsed_dict = parse_action_to_structure_output(
        response,
        model_type="qwen25vl" if local_mode else "", # 火山 API 模式下返回的格式是 <point> 而且坐标也不一样
        # 只有在 model_type 为非 qwen25vl 时有用
        factor=1000, 
        # 只有在 model_type 为 qwen25vl 时有用
        origin_resized_height=resized_height, 
        origin_resized_width=resized_width,
    )

    # update action inputs
    for key, value in parsed_dict[0]['action_inputs'].items():
        if key in ['start_box', 'end_box']:
            value = ast.literal_eval(value)
            if isinstance(value, list):
                updated_list = []
                for item_idx, item in enumerate(value):
                    if isinstance(item, float) and 0 <= item <= 1:
                        if item_idx % 2 == 0:
                            item *= resized_width / resize_factor
                        else:
                            item *= resized_height / resize_factor

                    updated_list.append(item)
                parsed_dict[0]['action_inputs'][key] = updated_list
        
    result = {
        "reflection": parsed_dict[0]['reflection'],
        "thought": parsed_dict[0]['thought'],
        "function": parsed_dict[0]['action_type'],
        "args": parsed_dict[0]['action_inputs'],
    }

    return result

def execute_ui_tars_response(request: dict, execute_resize_factor: float=1, serial: str='emulator-5554') -> UiTarsResult:
    '''
    execute_resize_factor: 除以该倍数以获得原始的坐标。parsed_response 中已经处理了 resize，因此这里一般不需要再进行 resize。
    '''
    if need_breakpoint: breakpoint()
    executed_successfully = False
    finished = False
    content = ''
    match request.get('function'):
        case 'click':
            try:
                x, y = int(request.get("args")['start_box'][0] / execute_resize_factor), int(request.get("args")['start_box'][1]  / execute_resize_factor)
                print(f"try to click on {x}, {y}")
                # click on the position
                duration = random.randint(200, 400)
                print(f"random duration: {duration}")
                # if random.randint(0, 1) % 2 == 0:
                #     print(f"choose swipe policy for click")
                #     os.system(f"adb -s {serial} shell input swipe {x} {y} {x} {y} {duration}")
                # else:
                print(f"choose tap policy for click")
                os.system(f"adb -s {serial} shell input tap {x} {y}")
                executed_successfully = True
                finished = False
            except Exception as e:
                print(f"Click failed: {e}")
                executed_successfully = False
                
        case 'long_press':
            try:
                x, y = int(request.get("args")['start_box'][0] / execute_resize_factor), int(request.get("args")['start_box'][1]  / execute_resize_factor)
                # long press on the position (longer duration)
                duration = 1000  # 1 second for long press
                os.system(f"adb -s {serial} shell input swipe {x} {y} {x} {y} {duration}")
                executed_successfully = True
                finished = False
            except Exception as e:
                print(f"Long press failed: {e}")
                executed_successfully = False
                
        case 'type':
            try:
                # os.system(f'adb -s {serial} shell input text "{text}"')
                # 检查输入文本是否以换行符结尾
                content: str = request.get("args").get('content', '')
                has_trailing_newline = content.endswith("\n")

                # 将字符串按照换行符拆分
                lines = content.splitlines()

                # 遍历所有行并输入文本
                for i, line in enumerate(lines):
                    # 将空格替换为ADB要求的转义字符 %s
                    escaped_line = line.replace(" ", "%s")
                    escaped_line = escaped_line.replace("'", "\\'")
                    os.system(f'adb -s {serial} shell input text "{escaped_line}"')
                    # 如果不是最后一行，则模拟回车（KEYCODE_ENTER对应66）
                    if i < len(lines) - 1:
                        os.system(f'adb -s {serial} shell input keyevent 66')

                # 如果输入的文本最后有换行符，则额外模拟一次回车
                if has_trailing_newline:
                    os.system(f'adb -s {serial} shell input keyevent 66')
                executed_successfully = True

            except Exception as e:
                print(f"Type failed: {e}")
                executed_successfully = False
                
        case 'scroll':
            try:
                x, y = int(request.get("args")['start_box'][0] / execute_resize_factor), int(request.get("args")['start_box'][1]  / execute_resize_factor)
                direction = request.get("args").get('direction', 'down')
                
                # Define scroll distances
                scroll_distance = 500
                
                if direction == 'down':
                    end_x, end_y = x, y - scroll_distance
                elif direction == 'up':
                    end_x, end_y = x, y + scroll_distance
                elif direction == 'left':
                    end_x, end_y = x + scroll_distance, y
                elif direction == 'right':
                    end_x, end_y = x - scroll_distance, y
                else:
                    raise ValueError(f"Invalid scroll direction: {direction}")
                
                duration = 500
                os.system(f"adb -s {serial} shell input swipe {x} {y} {end_x} {end_y} {duration}")
                executed_successfully = True
                finished = False
            except Exception as e:
                print(f"Scroll failed: {e}")
                executed_successfully = False
                
        case 'open_app':
            try:
                app_name = request.get("args").get('app_name', '')
                if not app_name:
                    raise ValueError("App name is required")
                
                # Try to launch app by package name or activity
                start_app(app_name, serial)
                time.sleep(2)  # Wait for app to launch
                executed_successfully = True
                finished = False
            except Exception as e:
                print(f"Open app failed: {e}")
                executed_successfully = False
                
        case 'drag':
            try:
                start_x, start_y = int(request.get("args")['start_box'][0] / execute_resize_factor), int(request.get("args")['start_box'][1] / execute_resize_factor)
                end_x, end_y = int(request.get("args")['end_box'][0] / execute_resize_factor), int(request.get("args")['end_box'][1] / execute_resize_factor)
                
                duration = 800  # Longer duration for drag
                os.system(f"adb -s {serial} shell input swipe {start_x} {start_y} {end_x} {end_y} {duration}")
                executed_successfully = True
                finished = False
            except Exception as e:
                print(f"Drag failed: {e}")
                executed_successfully = False
                
        case 'press_home':
            try:
                # Press home button
                os.system(f"adb -s {serial} shell input keyevent 3")  # KEYCODE_HOME
                executed_successfully = True
                finished = False
            except Exception as e:
                print(f"Press home failed: {e}")
                executed_successfully = False
                
        case 'press_back':
            try:
                # Press back button
                os.system(f"adb -s {serial} shell input keyevent 4")  # KEYCODE_BACK
                executed_successfully = True
                finished = False
            except Exception as e:
                print(f"Press back failed: {e}")
                executed_successfully = False
                
        case 'finished':
            try:
                content = request.get("args", {}).get('content', 'Task completed')
                print(f"Task finished: {content}")
                executed_successfully = True
                finished = True
            except Exception as e:
                print(f"Finished failed: {e}")
                executed_successfully = False
                finished = True
                
        case _:
            print(f"Unknown function: {request.get('function')}")
            executed_successfully = False
            finished = False

    return UiTarsResult(
        executed_successfully=executed_successfully,
        finished=finished,
        content=content
    )

def ui_tars_work_loop(local_mode: bool=True, get_ui_tars_response=None, get_ui_tars_mobile_prompt=None, language: str='Chinese', instruction: str=None, get_screenshot: Callable[[str], Image.Image]=take_screenshot_adb, image_resize_factor=0.5, serial='emulator-5554') -> UiTarsLoopResult:
    '''
    please use ui_tars15_work_loop instead.
    local_mode: 请求本地的 api。由于 qwen2.5vl 的 message 序列化方式与 OpenAI 有些微的不同，需要做一点修改。目前 openai 格式的后端应该是兼容 qwen2.5vl 的后端，但是反过来不行。
    ui_tars 的火山版本似乎与本地版本不同，官方发布的 action_parser 也不支持火山的版本，因此目前先考虑用本地版本来进行操作。
    '''
    serialize_mode = 'qwen' if local_mode else 'openai'
    get_ui_tars_response = get_ui_tars_response or init_get_ui_tars_response()
    get_ui_tars_mobile_prompt = get_ui_tars_mobile_prompt or (get_ui_tars_mobile_prompt_local if local_mode else get_ui_tars_mobile_prompt_api)
    history_list = []
    for _ in range(UI_TARS_MAX_LOOP):
        time.sleep(1)
        current_screenshot = get_screenshot(serial=serial)
        resized_current_screenshot = current_screenshot.resize((round(current_screenshot.width * image_resize_factor), round(current_screenshot.height * image_resize_factor)))
        for request_attempt in tenacity.Retrying(stop=tenacity.stop_after_attempt(3), before=lambda s: logger.info(f"Attempt {s.attempt_number} to request ui tars", instruction=instruction), after=log_retry_error_with_traceback):
            with request_attempt:
                messages = get_ui_tars_messages(
                    get_ui_tars_mobile_prompt=get_ui_tars_mobile_prompt, 
                    language=language, 
                    instruction=instruction, 
                    screenshot=resized_current_screenshot,
                    history_list=history_list,
                    serialize_mode=serialize_mode
                )
                response = get_ui_tars_response(messages)
                print(response)

                for parse_attempt in tenacity.Retrying(stop=tenacity.stop_after_attempt(10), before=lambda s: logger.info(f"Attempt {s.attempt_number} to parse response", response=response), after=log_retry_error_with_traceback):
                    with parse_attempt:
                        if parse_attempt.retry_state.attempt_number > 1:
                            get_response = init_get_openai_response()
                            fixed_response = get_response([make_user(get_fix_prompt(response)).serialize()])
                            parsed_dict = parse_ui_tars_response(local_mode, fixed_response, resized_current_screenshot.width, resized_current_screenshot.height, image_resize_factor)
                        else:
                            parsed_dict = parse_ui_tars_response(local_mode, response, resized_current_screenshot.width, resized_current_screenshot.height, image_resize_factor)
                        
        clear_and_print_pairs(("instruction", instruction), ("response:", response), ("parsed dict:", str(parsed_dict)))
        if need_breakpoint: breakpoint()
        
        execution_result = execute_ui_tars_response(parsed_dict)
        if execution_result.finished:
            print(f"execution completed! feedback: {execution_result.content}, task: {instruction}")
            return UiTarsLoopResult(
                content=execution_result.content,
                finished=True,
                error_info=''
            )
        history_list.append((resized_current_screenshot, response))
        time.sleep(1)
    print(f"execution reached max loop time! feedback: {execution_result.content}, task: {instruction}")
    return UiTarsLoopResult(
        content=execution_result.content,
        finished=False,
        error_info='Reached Max Loop Time'
    )

if __name__ == "__main__":
    import json
    from pathlib import Path
    from PIL import Image, ImageDraw
    from agentprog.all_utils.general_utils import Message, Messages, Prompt, make_user, init_get_openai_response, image_to_base64, init_get_claude_response
    from agentprog.all_utils.ui_tars15 import ui_tars15_work_loop
    from agentprog.all_utils.ui_tars_utils import get_ui_tars_mobile_prompt_local, init_get_ui_tars_response, parse_ui_tars_response, ui_tars_work_loop, get_ui_tars_mobile_prompt_api, take_screenshot_adb, start_app
    from ui_tars.action_parser import convert_point_to_coordinates, smart_resize
    import matplotlib.pyplot as plt
    import os
    get_ui_tars_response = init_get_ui_tars_response()
    # get_ui_tars_response = init_get_ui_tars_response(
    #     base_url='http://127.0.0.1:8888',
    #     api_key=''
    # )
    # result = ui_tars_work_loop(
    #     local_mode=True,
    #     get_ui_tars_response=get_ui_tars_response,
    #     get_ui_tars_mobile_prompt=get_ui_tars_mobile_prompt_api,
    #     language="Chinese",
    #     instruction="""Record an audio in audio recorder.""",
    #     get_screenshot=take_screenshot_adb,
    # )
    # result = ui_tars15_work_loop(
    #     instruction='Record an audio in audio recorder.',
    #     serial='emulator-5556'
    # )
    # print(result)


    fix_prompt = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 

## Output Format
```
Thought: ...
Action: ...
```

## Action Space

click(point='<point>x1 y1</point>')
long_press(point='<point>x1 y1</point>')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(point='<point>x1 y1</point>', direction='down or up or right or left')
open_app(app_name=\'\')
drag(start_point='<point>x1 y1</point>', end_point='<point>x2 y2</point>')
press_home()
press_back()
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.

"""
    res: RichStr = get_ui_tars_response([make_user(fix_prompt).serialize()])
    print(res.get_raw())