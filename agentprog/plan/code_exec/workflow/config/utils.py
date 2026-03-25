from __future__ import annotations
import json
from argparse import ArgumentParser, Namespace
from agentprog.plan.agentprog_utils import RequestMode, ToolSet

str_to_bool = lambda s: True if s.lower() == "true" else False

def resolve_model_args(full_args: Namespace):
    """Collect --{prefix}.field dot-style args into InitResponseArgs objects and remove the flat keys."""
    from agentprog.all_utils.general_utils import InitResponseArgs
    for prefix in ("workflow_model_args", "executor_model_args"):
        flat_keys = [k for k in list(vars(full_args)) if k.startswith(f"{prefix}__")]
        kwargs = {}
        for k in flat_keys:
            value = vars(full_args).pop(k)
            if value is not None:
                field_name = k[len(prefix) + 2:]
                kwargs[field_name] = json.loads(value) if field_name == "completion_kwargs" else value
        if kwargs:
            vars(full_args)[prefix] = InitResponseArgs(**kwargs)

def add_common_args(arg_parser: ArgumentParser):
    arg_parser.add_argument("--request_mode", default=None, choices=RequestMode._member_names_, required=False, help="Type of agent_prog mode (api or local llm)")
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
    arg_parser.add_argument("--show_dashboard", default=None, type=str_to_bool, required=False, help="Show the dashboard")
    arg_parser.add_argument("--fold_dashboard", default=None, type=str_to_bool, required=False, help="Fold the dashboard")
    for prefix in ("workflow_model_args", "executor_model_args"):
        arg_parser.add_argument(f"--{prefix}.model", dest=f"{prefix}__model", default=None, type=str, required=False, help=f"Model name for {prefix} in LiteLLM style (e.g. gemini/gemini-2.5-pro)")
        arg_parser.add_argument(f"--{prefix}.base_url", dest=f"{prefix}__base_url", default=None, type=str, required=False, help=f"Base URL for {prefix}")
        arg_parser.add_argument(f"--{prefix}.api_key", dest=f"{prefix}__api_key", default=None, type=str, required=False, help=f"API key for {prefix}")
        arg_parser.add_argument(f"--{prefix}.use_sdk", dest=f"{prefix}__use_sdk", default=None, type=str_to_bool, required=False, help=f"Use SDK (vs requests) for {prefix}")
        arg_parser.add_argument(f"--{prefix}.completion_kwargs", dest=f"{prefix}__completion_kwargs", default=None, type=str, required=False, help=f'JSON string of extra completion kwargs for {prefix} (e.g. \'{{"temperature": 0.5}}\')')
