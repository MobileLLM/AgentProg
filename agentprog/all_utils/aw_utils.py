from __future__ import annotations
import ast
from dataclasses import dataclass
import os
import re
import time
from PIL import Image
import numpy as np
from android_world.env import interface
from android_world.agents.m3a import _generate_ui_element_description, _generate_ui_elements_description_list, _action_selection_prompt, m3a_utils
from android_world.env import env_launcher

import structlog

from agentprog.all_utils.fm import FoundationModel
from agentprog.all_utils.mobile_utils import LocatorAPI, MobileAPI, MobileAPIConfig

from agentprog.all_utils import log_utils
logger = log_utils.get_logger(__name__)

@dataclass
class ScreenDescription:
    state: interface.State
    before_ui_elements_list: list
    before_screenshot: np.ndarray
    before_screenshot_with_som: np.ndarray

def _prepare_grounding_prompt(target_description, screen_description: ScreenDescription):
    return f'''
You are an agent who can operate an Android phone on behalf of a user. 
Based on user's goal/request, you may complete some tasks described in the requests/goals by performing actions (step by step) on the phone.

At each step, you will be given the current screenshot (including the original screenshot and the same screenshot with bounding boxes and numeric indexes added to some UI elements) and a history of what you have done (in text). Based on these pieces of information and the goal, you should output the correct action in the correct JSON format.

You should click/tap on an element on the screen. We have added marks (bounding boxes with numeric indexes on their TOP LEFT corner) to most of the UI elements in the screenshot, use the numeric index to indicate which element you want to click, and output -1 if none matched.

The current user goal/request is: click on the view that matches description `{target_description}`.

The current screenshot and the same screenshot with bounding boxes and labels added are also given to you.\n

Special attention: Some target views may be enclosed by two boxes, but you should prioritize selecting the one that most accurately represents the target view, particularly by ensuring that:

1. The center of the box coincides with the center of the target view.
2. The edges of the box align with the boundaries of the target view.


''', "\nThe Current Screenshot: ", Image.fromarray(screen_description.before_screenshot), "\nThe labeled Currrent Screenshot: ", Image.fromarray(screen_description.before_screenshot_with_som)

def _prepare_check_prompt(target_description, screen_description: ScreenDescription):
    return f'''
You are a GUI agent. You are given a query and an answer format, with screenshots. Your task is to analyze the screenshots and provide the information requested in the query, strictly following the given answer format.\nQuery: Verify the following statements: {target_description}\nAnswer Format: true/false\nScreenshots: 
''', Image.fromarray(screen_description.before_screenshot)

@dataclass
class AndroidWorldLocatorConfig:
    serial_port: int
    grpc_port: int

class AndroidWorldLocator(LocatorAPI):
    '''
    尚未测试。
    '''
    def __init__(self, config: MobileAPIConfig, mobile_api: MobileAPI):
        self.config = config
        self.mobile_api = mobile_api
        assert self.config.llm is not None, "You should give a llm to use this locator"
        self.llm: FoundationModel = config.llm
        self.locator_config = config.locator_config
        assert isinstance(self.locator_config, AndroidWorldLocatorConfig), "You should set a android world locator config to use this locator"
        self.env: interface.AsyncEnv = env_launcher._get_env(
                console_port=int(self.locator_config.serial_port),
                adb_path=f"{os.environ['ANDROID_SDK_ROOT']}/platform-tools/adb",
                grpc_port=int(self.locator_config.grpc_port)
            )
        self._transition_pause: float | None = 1.0

    # llm
    def locate(self, description):
        """
        locate the view using an LLM
        """
        parent_view_desc = self._describe_screen()
        try:
            matched_idx = self.llm.query(
                *_prepare_grounding_prompt(description, parent_view_desc),
                returns=('the matched view index in int, -1 if none matched', int)
            )
            matched_idx = int(matched_idx)
            screen_elements = parent_view_desc.state.ui_elements
            if matched_idx < 0:
                raise RuntimeError(f"在当前界面定位 \"{description}\" 失败。如果 \"{description}\" 在当前界面存在，请点击 \"手工补丁\" 按钮添加补丁。")
            if matched_idx >= len(screen_elements):
                raise ValueError(
                    f'Invalid element index: {matched_idx}, must be between 0 and'
                    f' {len(screen_elements)-1}.'
                )
            element = screen_elements[matched_idx]
            if element.bbox_pixels is None:
                raise ValueError('Bbox is not present on element.')

            x, y = element.bbox_pixels.center
            x, y = int(x), int(y)
            return (x, y, x, y), matched_idx
            
        except Exception as e:
            logger.exception(f'{self._tag} _locate_view exception {e}', action='locate view', status='failed')
        return (None, None, None, None), -1

    def check(self, description: str | list[str]):
        parent_view_desc = self._describe_screen()

        check_result = self.llm.query(*_prepare_check_prompt(description, parent_view_desc), returns=("is the statement correct?", bool))
        
        logger.info(f"🔎 检查 \"{description}\"，结果为：{check_result}")
        return check_result
    
    def _describe_screen(self) -> ScreenDescription:
        """
        get a tree description of the given view
        """
        step_data = {
            'raw_screenshot': None,
            'before_screenshot_with_som': None,
            'before_ui_elements': [],
            'after_screenshot_with_som': None,
            'action_prompt': None,
            'action_output': None,
            'action_output_json': None,
            'action_reason': None,
            'action_raw_response': None,
            'summary_prompt': None,
            'summary': None,
            'summary_raw_response': None,
        }
        state = self._get_post_transition_state()
        logical_screen_size = self.env.logical_screen_size
        orientation = self.env.orientation
        physical_frame_boundary = self.env.physical_frame_boundary

        before_ui_elements = state.ui_elements
        step_data['before_ui_elements'] = before_ui_elements
        before_ui_elements_list = _generate_ui_elements_description_list(
            before_ui_elements, logical_screen_size
        )
        step_data['raw_screenshot'] = state.pixels.copy()
        before_screenshot = state.pixels.copy()
        for index, ui_element in enumerate(before_ui_elements):
            if m3a_utils.validate_ui_element(ui_element, logical_screen_size):
                m3a_utils.add_ui_element_mark(
                    before_screenshot,
                    ui_element,
                    index,
                    logical_screen_size,
                    physical_frame_boundary,
                    orientation,
                )
        step_data['before_screenshot_with_som'] = before_screenshot.copy()

        return ScreenDescription(
            state=state,
            before_ui_elements_list=before_ui_elements_list,
            before_screenshot=step_data['raw_screenshot'],
            before_screenshot_with_som=step_data['before_screenshot_with_som']
        )

    def _get_post_transition_state(self) -> interface.State:
        """Convenience function to get the agent state after the transition."""
        if self._transition_pause is None:
            print(
                'Waiting for screen to stabilize before grabbing state...',
                end=' ',
            )
            start = time.time()
            state = self.env.get_state(wait_to_stabilize=True)
            print(f'Fetched after {time.time() - start:2.1f} seconds.')
            return state
        else:
            time.sleep(self._transition_pause)
            print(
                'Pausing {:2.1f} seconds before grabbing state.'.format(
                    self._transition_pause
                )
            )
            return self.env.get_state(wait_to_stabilize=False)
