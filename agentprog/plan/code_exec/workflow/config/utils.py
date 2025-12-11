from __future__ import annotations
from argparse import ArgumentParser
from agentprog.plan.agentprog_utils import RequestMode, ToolSet

str_to_bool = lambda s: True if s.lower() == "true" else False

def add_common_args(arg_parser: ArgumentParser):
    arg_parser.add_argument("--request_mode", default=None, choices=RequestMode._member_names_, required=False, help="Type of interpreterllm mode (api or local llm)")
    arg_parser.add_argument("--tool_set", default=None, choices=ToolSet._member_names_, required=False, help="Whether to use tools or interact with environment (mobile, ...)") # 未来也许要扩展成 mcp 那种形式，可以自由地加减工具和环境。
    arg_parser.add_argument("--image_dir", default=None, type=str, required=False, help="Path to save screenshot")
    arg_parser.add_argument("--meta_info_dir", default=None, type=str, required=False, help="Path to save workflow meta info")
    arg_parser.add_argument("--serial", default=None, type=str, required=False, help="Adb Serial Name")
    arg_parser.add_argument("--serial_port", default=None, type=str, required=False, help="Adb Serial Port, Matched to Serial Name") # 这和上面那个不是严格对应的，但是我们希望上一层至少能传一个下来，这样我们至少能根据这两个中的一个相互进行查找。
    arg_parser.add_argument("--cache_mode", default=None, type=str_to_bool, required=False, help="Use cache mode to reuse the code of executed node")
    arg_parser.add_argument("--use_belief_state", default=None, type=str_to_bool, required=False, help="Use Belief State to Enhance Interaction")
    arg_parser.add_argument("--use_aw_locator", default=None, type=str_to_bool, required=False, help="use android world locator")
    arg_parser.add_argument("--tensorboard_log_dir", default=None, type=str, required=False, help="tensorboard logs")
    arg_parser.add_argument("--logging_path", default=None, type=str, required=False, help="logging path")
