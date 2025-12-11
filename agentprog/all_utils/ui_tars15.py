"""UITARS15 Agent implementation for Android World evaluation framework."""

import ast
import base64
import io
import json
import logging
import math
import os
from pathlib import Path
import re
import time
import traceback
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Any, Callable, Optional, List, Dict, Tuple, Union
from PIL import Image
import numpy as np
import requests
import urllib

from tenacity import retry, stop_after_attempt, wait_random_exponential
import tenacity

from agentprog.all_utils.ui_tars_utils import UiTarsLoopResult, clear_and_print_pairs, execute_ui_tars_response, need_breakpoint, take_screenshot_adb
from agentprog.all_utils.general_utils import log_retry_error_with_traceback

# from android_world.agents import base_agent
# from android_world.attack.nodes import capture_action
# from android_world.env import interface
# from android_world.env import json_action
# from android_world.env import representation_utils
# from android_world.env.android_world_controller import A11yMethod

# UITARS15 Constants
FINISH_WORD = "finished"
WAIT_WORD = "wait"
ENV_FAIL_WORD = "error_env"
CALL_USER = "call_user"

IMAGE_FACTOR = 28
MIN_PIXELS = 100 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

# Mobile action space prompts for UITARS15
MOBILE_USE_WITH_THINKING = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
You should first think about the reasoning process in the mind and then provide the user with the answer. 
The reasoning process is enclosed within <think> </think> tags
After the <think> tags, you should place final answer, which concludes your summarized thought and your action.

For example,
```
<think>detailed reasoning content here</think>
Thought: a small plan and finally summarize your next action (with its target element) in one sentence
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

## Output Example
<think>Now that...</think>
Thought: Let's click ...
Action: click(point='<point>100 200</point>')

## Note
- Use {language} in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.
- If you have executed several same actions (like repeatedly clicking the same point) but the screen keeps no change, please try to execute a modified action when necessary.

## User Instruction
{instruction}
"""

MOBILE_USE_WITHOUT_THINKING = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 

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
"""

logger = logging.getLogger("android_world.uitars15")
logger.setLevel(logging.INFO)

def image2PIL(image: np.ndarray) -> Image:
    """Convert numpy array to PIL image."""
    image = Image.fromarray(image).convert("RGB")
    return image


def pil_to_base64(image: Image.Image) -> str:
    """Convert PIL image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def convert_point_to_coordinates(text, is_answer=False):
    """Convert point format to coordinates."""
    pattern = r"<point>(\d+)\s+(\d+)</point>"
    
    def replace_match(match):
        x1, y1 = map(int, match.groups())
        x = (x1 + x1) // 2
        y = (y1 + y1) // 2
        if is_answer:
            return f"({x},{y})"
        return f"({x},{y})"
    
    text = re.sub(r"\[EOS\]", "", text)
    return re.sub(pattern, replace_match, text).strip()


def parse_action(action_str: str) -> Optional[Dict[str, Any]]:
    """Parse action string into structured format."""
    try:
        node = ast.parse(action_str, mode='eval')
        if not isinstance(node, ast.Expression):
            raise ValueError("Not an expression")

        call = node.body
        if not isinstance(call, ast.Call):
            raise ValueError("Not a function call")

        # Get function name
        if isinstance(call.func, ast.Name):
            func_name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            func_name = call.func.attr
        else:
            func_name = None

        # Get keyword arguments
        kwargs = {}
        for kw in call.keywords:
            key = kw.arg
            if isinstance(kw.value, ast.Constant):
                value = kw.value.value
            elif isinstance(kw.value, ast.Str):  # Compatibility with older Python
                value = kw.value.s
            else:
                value = None
            kwargs[key] = value

        return {
            'function': func_name,
            'args': kwargs
        }
    except Exception as e:
        logger.error(f"Failed to parse action '{action_str}': {e}")
        return None


def escape_single_quotes(text: str) -> str:
    """Escape single quotes in text."""
    pattern = r"(?<!\\)'"
    return re.sub(pattern, r"\\'", text)


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def linear_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """Resize image maintaining aspect ratio within pixel limits."""
    if width * height > max_pixels:
        resize_factor = math.sqrt(max_pixels / (width * height))
        width, height = int(width * resize_factor), int(height * resize_factor)
    if width * height < min_pixels:
        resize_factor = math.sqrt(min_pixels / (width * height))
        width, height = math.ceil(width * resize_factor), math.ceil(height * resize_factor)
    return height, width


def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """Smart resize maintaining aspect ratio and factor divisibility."""
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, "
            f"got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def parse_action_to_structure_output(
    text: str, factor: int, origin_resized_height: int, origin_resized_width: int,
    model_type: str = "doubao", max_pixels: int = 16384*28*28, min_pixels: int = 100*28*28
) -> List[Dict[str, Any]]:
    """Parse UITARS15 response text into structured actions."""
    text = text.strip()

    # Convert point format to coordinates
    if "<point>" in text:
        text = convert_point_to_coordinates(text)
    if "start_point=" in text:
        text = text.replace("start_point=", "start_box=")
    if "end_point=" in text:
        text = text.replace("end_point=", "end_box=")
    if "point=" in text:
        text = text.replace("point=", "start_box=")

    if model_type == "qwen25vl":
        smart_resize_height, smart_resize_width = smart_resize(
            origin_resized_height, origin_resized_width,
            factor=IMAGE_FACTOR, min_pixels=min_pixels, max_pixels=max_pixels
        )

    # Extract thought and action
    thought_pattern = r"Thought: (.+?)(?=\s*Action:|$)"
    reflection, thought = None, None
    thought_match = re.search(thought_pattern, text, re.DOTALL)
    if thought_match:
        thought = thought_match.group(1).strip()

    if "Action:" not in text:
        raise ValueError("No Action found in response")

    action_str = text.split("Action:")[-1].strip()

    # Handle multiple actions
    tmp_all_action = action_str.split("')\n\n")
    all_action = []
    for action_str in tmp_all_action:
        if "type(content" in action_str:
            # Handle special case for type action
            def escape_quotes(match):
                return match.group(1)
            pattern = r"type\(content='(.*?)'\)"
            content = re.sub(pattern, escape_quotes, action_str)
            action_str = escape_single_quotes(content)
            action_str = "type(content='" + action_str + "')"
        all_action.append(action_str)

    parsed_actions = [parse_action(action.replace("\n", "\\n").lstrip()) for action in all_action]
    actions = []

    for action_instance, raw_str in zip(parsed_actions, all_action):
        if action_instance is None:
            logger.error(f"Action can't parse: {raw_str}")
            raise ValueError(f"Action can't parse: {raw_str}")

        action_type = action_instance["function"]
        params = action_instance["args"]

        action_inputs = {}
        for param_name, param in params.items():
            if param == "":
                continue
            param = str(param).lstrip()
            action_inputs[param_name.strip()] = param

            # Handle start_box or end_box parameters
            if "start_box" in param_name or "end_box" in param_name:
                ori_box = param
                numbers = ori_box.replace("(", "").replace(")", "").split(",")

                if model_type == "qwen25vl":
                    float_numbers = []
                    for num_idx, num in enumerate(numbers):
                        num = float(num)
                        if (num_idx + 1) % 2 == 0:
                            float_numbers.append(float(num/smart_resize_height))
                        else:
                            float_numbers.append(float(num/smart_resize_width))
                else:
                    float_numbers = [float(num) / factor for num in numbers]

                if len(float_numbers) == 2:
                    float_numbers = [float_numbers[0], float_numbers[1], float_numbers[0], float_numbers[1]]
                action_inputs[param_name.strip()] = str(float_numbers)

    # update action inputs
    # 从而转换为绝对坐标
    for key, value in action_inputs.items():
        if key in ['start_box', 'end_box']:
            value = ast.literal_eval(value)
            if isinstance(value, list):
                updated_list = []
                for item_idx, item in enumerate(value):
                    if isinstance(item, float) and 0 <= item <= 1:
                        if item_idx % 2 == 0:
                            item *= origin_resized_width
                        else:
                            item *= origin_resized_height

                    updated_list.append(item)
                action_inputs[key] = updated_list
        
    result = {
        "reflection": reflection,
        "thought": thought,
        "function": action_type,
        "args": action_inputs,
    }

    return result


def get_fix_prompt(response: str):
    return f'''
允许调用的 API 为：

click(point='<point>x1 y1</point>')
long_press(point='<point>x1 y1</point>')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(point='<point>x1 y1</point>', direction='down or up or right or left')
open_app(app_name=\'\')
drag(start_point='<point>x1 y1</point>', end_point='<point>x2 y2</point>')
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

def uitars15_action_to_android_action(
    parsed_response: Dict[str, Any], screen_size: Tuple[int, int]
) -> Optional[Dict[str, Any]]:
    """Convert UITARS15 action to android_world JSON action format."""
    result_action = {
        "action_type": None,
        "x": None,
        "y": None,
        "direction": None,
        "text": None,
        "app_name": None,
    }

    action_type = parsed_response.get("action_type")
    action_inputs = parsed_response.get("action_inputs", {})

    # Handle different UITARS15 action types (mobile-specific)
    if action_type == "click":
        result_action["action_type"] = "click"
    elif action_type == "long_press":
        result_action["action_type"] = "long_press"
    elif action_type == "type":
        result_action["action_type"] = "input_text"
        result_action["text"] = action_inputs.get("content", "")
    elif action_type == "scroll":
        result_action["action_type"] = "scroll"
        direction = action_inputs.get("direction", "down")
        result_action["direction"] = direction.lower()
    elif action_type == "drag":
        result_action["action_type"] = "drag"
    elif action_type == "press_home":
        result_action["action_type"] = "navigate_home"
    elif action_type == "press_back":
        result_action["action_type"] = "navigate_back"
    elif action_type == "open_app":
        result_action["action_type"] = "open_app"
        result_action["app_name"] = action_inputs.get("app_name", "")
    elif action_type in [FINISH_WORD, "finished"]:
        return {"action_type": "status", "goal_status": "complete"}
    elif action_type in [WAIT_WORD, "wait"]:
        return {"action_type": "wait"}
    else:
        logger.warning(f"Unsupported action type: {action_type}")
        return None

    # Handle coordinates from start_box
    if "start_box" in action_inputs:
        try:
            box_coords = eval(action_inputs["start_box"])
            if len(box_coords) >= 2:
                x1, y1 = box_coords[0], box_coords[1]
                if len(box_coords) >= 4:
                    x2, y2 = box_coords[2], box_coords[3]
                    x, y = (x1 + x2) / 2, (y1 + y2) / 2
                else:
                    x, y = x1, y1

                # Convert relative coordinates to absolute
                result_action["x"] = int(x * screen_size[0])
                result_action["y"] = int(y * screen_size[1])
        except Exception as e:
            logger.error(f"Error parsing start_box coordinates: {e}")
            return None

    # Handle drag end coordinates
    if action_type == "drag" and "end_box" in action_inputs:
        try:
            end_coords = eval(action_inputs["end_box"])
            if len(end_coords) >= 2:
                x1, y1 = end_coords[0], end_coords[1]
                if len(end_coords) >= 4:
                    x2, y2 = end_coords[2], end_coords[3]
                    end_x, end_y = (x1 + x2) / 2, (y1 + y2) / 2
                else:
                    end_x, end_y = x1, y1

                result_action["end_x"] = int(end_x * screen_size[0])
                result_action["end_y"] = int(end_y * screen_size[1])
        except Exception as e:
            logger.error(f"Error parsing end_box coordinates: {e}")

    return result_action


class UITARS15Agent:
    """UITARS15 Agent for Android World evaluation framework.

    This agent supports mobile action space only and includes both reasoning modes:
    - With thinking (explicit reasoning steps)
    - Without thinking (direct reasoning)
    """

    def __init__(
        self,
        name: str = "uitars15",
        wait_after_action_seconds: float = 2.0,
        model_type: str = "doubao",
        use_thinking: bool = True,
        language: str = "English",
        runtime_conf: Optional[Dict[str, Any]] = None,
        meta_info_dir: str=None
    ):
        """Initialize UITARS15 Agent.

        Args:
            env: The Android World environment
            name: Agent name
            wait_after_action_seconds: Seconds to wait after executing an action
            model_type: Model type for coordinate processing
            use_thinking: Whether to use thinking mode (with <think> tags)
            language: Language for responses
            runtime_conf: Runtime configuration dictionary
        """

        # Default runtime configuration
        default_conf = {
            "max_tokens": 500,
            "temperature": 0.0,
            "top_p": 0.9,
            "history_n": 5,
            "max_pixels": 16384*28*28,
            "min_pixels": 100*28*28,
            "callusr_tolerance": 3,
        }

        self.runtime_conf = {**default_conf, **(runtime_conf or {})}
        self.model_type = model_type
        self.wait_after_action_seconds = wait_after_action_seconds
        self.use_thinking = use_thinking
        self.language = language
        self.is_misled = False

        # UITARS15 specific attributes
        self.thoughts = []
        self.actions = []
        self.observations = []
        self.history_images = []
        self.history_responses = []
        self.action_parse_res_factor = 1000
        self.cur_callusr_count = 0

        # Configuration shortcuts
        self.max_tokens = self.runtime_conf["max_tokens"]
        self.temperature = self.runtime_conf["temperature"]
        self.top_p = self.runtime_conf["top_p"]
        self.history_n = self.runtime_conf["history_n"]
        self.max_pixels = self.runtime_conf["max_pixels"]
        self.min_pixels = self.runtime_conf["min_pixels"]
        self.callusr_tolerance = self.runtime_conf["callusr_tolerance"]
        self.json_result = []
        self.meta_info_dir = Path(meta_info_dir) if meta_info_dir is not None else None
        self.response_json_path = self.meta_info_dir / "responses.json" if self.meta_info_dir is not None else None

        # Select prompt based on thinking mode
        if use_thinking:
            self.system_prompt = MOBILE_USE_WITH_THINKING
        else:
            self.system_prompt = MOBILE_USE_WITHOUT_THINKING

    def reset(self, go_home_on_reset: bool = False):
        """Reset the agent state."""
        # Hide the coordinates on screen which might affect the vision model
        self.is_misled = False

        # Reset UITARS15 specific state
        self.thoughts = []
        self.actions = []
        self.observations = []
        self.history_images = []
        self.history_responses = []
        self.cur_callusr_count = 0

    def _call_llm(self, messages: List[Dict[str, Any]]) -> str:
        """Call the LLM API with the specified format."""
        api_key = os.environ.get('ARK_API_KEY')
        api_url = os.environ.get('DOUBAO_BASE_URL') + "/chat/completions"

        if not api_key or not api_url:
            raise ValueError("DOUBAO_API_KEY and DOUBAO_API_URL environment variables must be set")

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        data = {
            # "model": "ep-20250722213820-fqrbm",
            "model": "doubao-1.5-ui-tars-250428",
            "messages": messages,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "temperature": self.temperature,
        }

        # Add thinking configuration based on mode
        if self.use_thinking:
            data["thinking"] = {"type": "enabled"}
        else:
            data["thinking"] = {"type": "disabled"}

        response = requests.post(api_url, headers=headers, json=data)

        if response.status_code == 200:
            result = response.json()
            self.json_result.append(result)
            if self.response_json_path is not None:
                self.response_json_path.write_text(json.dumps(self.json_result, ensure_ascii=False))
            return result["choices"][0]["message"]["content"]
        else:
            logger.error(f"LLM API call failed with status {response.status_code}: {response.text}")
            raise Exception(f"LLM API call failed with status {response.status_code}")

    def pretty_print_messages(self, messages):
        """Pretty print messages while hiding base64 encoded images."""
        def format_message(msg):
            if not isinstance(msg, dict):
                return str(msg)

            formatted = {}
            for key, value in msg.items():
                if key == "content":
                    if isinstance(value, list):
                        formatted_content = []
                        for item in value:
                            if isinstance(item, dict) and "type" in item:
                                if item["type"] == "image_url" and "image_url" in item:
                                    # Replace base64 image with placeholder
                                    formatted_content.append({
                                        "type": "image_url",
                                        "image_url": {"url": "[BASE64_IMAGE_DATA]"}
                                    })
                                else:
                                    formatted_content.append(item)
                            else:
                                formatted_content.append(item)
                        formatted[key] = formatted_content
                    else:
                        formatted[key] = value
                else:
                    formatted[key] = value
            return formatted

        if isinstance(messages, list):
            return [format_message(msg) for msg in messages]
        return format_message(messages)

    @retry(stop=stop_after_attempt(10), wait=wait_random_exponential(max=60))
    def _predict_uitars15(self, instruction: str, obs: Dict[str, Any]) -> Tuple[str, List[str]]:
        """UITARS15 prediction method adapted for android_world.
        obs: {"screenshot": Image, "accessibility_tree": str}
        
        """

        # Append current observation to trajectory
        if len(self.observations) == len(self.actions) and len(self.actions) == len(self.thoughts):
            # Add observation
            self.observations.append({
                "screenshot": obs["screenshot"],
                "accessibility_tree": obs.get("accessibility_tree")
            })

        # Prepare user prompt
        user_prompt = self.system_prompt.format(
            instruction=instruction,
            language=self.language
        )

        # Limit history to recent images
        if len(self.history_images) > self.history_n:
            self.history_images = self.history_images[-self.history_n:]

        # Prepare messages for API
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}]
            },
        ]

        # Process images
        images = []
        for image_bytes in self.history_images[-self.history_n:]:
            try:
                image = Image.open(io.BytesIO(image_bytes))

                # Resize image if needed
                if image.width * image.height > self.max_pixels:
                    resize_factor = math.sqrt(self.max_pixels / (image.width * image.height))
                    width, height = int(image.width * resize_factor), int(image.height * resize_factor)
                    image = image.resize((width, height))
                if image.width * image.height < self.min_pixels:
                    resize_factor = math.sqrt(self.min_pixels / (image.width * image.height))
                    width, height = math.ceil(image.width * resize_factor), math.ceil(image.height * resize_factor)
                    image = image.resize((width, height))

                if image.mode != "RGB":
                    image = image.convert("RGB")

                images.append(image)
            except Exception as e:
                logger.error(f"Error processing image: {e}")
                continue

        # Add history responses and images to messages
        image_num = 0
        if len(self.history_responses) > 0:
            for history_idx, history_response in enumerate(self.history_responses):
                # Send at most history_n images to the model
                if history_idx + self.history_n > len(self.history_responses):
                    if image_num < len(images):
                        cur_image = images[image_num]
                        encoded_string = pil_to_base64(cur_image)
                        messages.append({
                            "role": "user",
                            "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_string}"}}]
                        })
                        image_num += 1

                messages.append({
                    "role": "assistant",
                    "content": history_response
                })

            # Add current image
            if image_num < len(images):
                cur_image = images[image_num]
                encoded_string = pil_to_base64(cur_image)
                messages.append({
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_string}"}}]
                })
        else:
            # First interaction - just add the current image
            if len(images) > 0:
                cur_image = images[-1]
                encoded_string = pil_to_base64(cur_image)
                messages.append({
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_string}"}}]
                })

        # Get prediction from model
        try_times = 3
        origin_resized_height = images[-1].height if images else None
        origin_resized_width = images[-1].width if images else None
        original_temperature = self.temperature
        for request_attempt in tenacity.Retrying(stop=tenacity.stop_after_attempt(3), before=lambda s: logger.info(f"Attempt {s.attempt_number} to request ui tars"), after=log_retry_error_with_traceback):
            with request_attempt:
                prediction = None
                while try_times > 0:
                    try:
                        # Update temperature for retry
                        self.temperature = original_temperature if try_times == 3 else 1.0
                        print(f"Messages: {self.pretty_print_messages(messages[-1])}")
                        if need_breakpoint: breakpoint()
                        prediction = self._call_llm(messages)
                        if prediction:
                            prediction = prediction.strip()
                            break
                    except Exception as e:
                        logger.error(f"Error when fetching response from client: {e}")
                        prediction = None
                        try_times -= 1

                # Restore original temperature
                self.temperature = original_temperature

                if prediction is None:
                    raise ValueError("Request Failed!")
                
                for parse_attempt in tenacity.Retrying(stop=tenacity.stop_after_attempt(3), before=lambda s: logger.info(f"Attempt {s.attempt_number} to parse response"), after=log_retry_error_with_traceback):
                    with parse_attempt:
                        # Parse the prediction
                        try:
                            print(prediction)
                            parsed_responses = parse_action_to_structure_output(
                                prediction,
                                self.action_parse_res_factor,
                                origin_resized_height,
                                origin_resized_width,
                                self.model_type,
                                self.max_pixels,
                                self.min_pixels
                            )
                        except Exception as e:
                            traceback.print_exc()
                            logger.error(f"Parsing action error: {prediction}, with error: {e}")
                            raise ValueError(f"Parsing action error: {prediction}, with error: {e}")
                        

                        # Store response in history
                        self.history_responses.append(prediction)
                        self.thoughts.append(prediction)
                        return parsed_responses

def ui_tars15_work_loop(instruction: str=None, serial='emulator-5554', get_screenshot: Callable[[str], Image.Image]=take_screenshot_adb, max_loop_time: int=50, image_resize_factor=1, use_thinking=False, img_save_dir="img_save_dir", meta_info_dir: str=None):
    img_save_dir = Path(img_save_dir)
    img_save_dir.mkdir(parents=True, exist_ok=True)
    meta_info_dir = Path(meta_info_dir)
    '''
    image_resize_factor: 缩小图片以减低请求开销。
    '''
    agent = UITARS15Agent(use_thinking=use_thinking, meta_info_dir=meta_info_dir)
    for step in range(max_loop_time):
        screenshot = get_screenshot(serial=serial)
        screenshot = screenshot.resize((round(screenshot.width * image_resize_factor), round(screenshot.height * image_resize_factor)))
        screenshot_bytes = io.BytesIO()
        screenshot.save(screenshot_bytes, format="PNG")
        screenshot.save(f"current_screenshot-{serial}.png")
        screenshot.save(img_save_dir / f"step_{step}.png")
        screenshot_bytes = screenshot_bytes.getvalue()
        agent.history_images.append(screenshot_bytes)
        obs = {"screenshot": screenshot_bytes, "accessibility_tree": ""}
        parsed_response = agent._predict_uitars15(instruction, obs)
        print(parsed_response)
        clear_and_print_pairs(("instruction", instruction), ("parsed dict:", str(parsed_response)))
        if need_breakpoint: breakpoint()
        execution_result = execute_ui_tars_response(parsed_response, image_resize_factor, serial)
        time.sleep(1)
        if execution_result.finished:
            print(f"execution completed! feedback: {execution_result.content}, task: {instruction}")
            return UiTarsLoopResult(
                content=execution_result.content,
                finished=True,
                error_info=''
            )
    
    return UiTarsLoopResult(
        content=execution_result.content,
        finished=False,
        error_info='Reached Max Loop Time'
    )
