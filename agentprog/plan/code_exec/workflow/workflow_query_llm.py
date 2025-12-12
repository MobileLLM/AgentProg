from __future__ import annotations
from enum import auto, Enum
from functools import partial
import itertools
import json
import os
import shlex
import shutil
import sys
from PIL import Image
from pathlib import Path
import time
from argparse import ArgumentParser, Namespace
import requests
from tenacity import retry, stop_after_attempt, wait_fixed

from agentprog.all_utils.mobile_utils import get_date_on_mobile_phone
from agentprog.plan.code_exec.general_prompts.framework_prompts.mobile_prompt import get_mobile_prompt
from agentprog.plan.code_exec.workflow.config.core_config import AgentProgConfig
from agentprog.plan.code_exec.workflow.config.utils import add_common_args
from agentprog.plan.code_exec.workflow.core.agentprog import run_workflow
from agentprog.all_utils.general_utils import IMAGE_END, IMAGE_START, TokenStatistics, dump_completion_statistics_dict, init_get_gemini_response, InitResponseArgs
from agentprog.plan.workflow_utils import WorkflowContext, HIDDEN_VARS_PREFIX
from agentprog.plan.agentprog_utils import AgentProgContext, RequestMode, ToolSet, filter_variables_list

from dotenv import load_dotenv
load_dotenv(override=True)

CLI_LEADER="#agentprog!"

def _agentprog_run_mobile(config: AgentProgConfig):
    from agentprog.all_utils.mobile_utils import get_text_description, MobileAPI, MobileAPIConfig
    from agentprog.all_utils.fm import get_default_fm
    SCREENSHOT_FUNC = f"{HIDDEN_VARS_PREFIX}_get_screenshot"
    SCREENSHOT_DICT = f"{HIDDEN_VARS_PREFIX}_screenshot_dict"
    get_response = init_get_gemini_response(init_response_args=InitResponseArgs(model='vertex_ai/gemini-2.5-pro', record_completion_statistics=True, token_budget=TokenStatistics(prompt_tokens=550000 * 6, completion_tokens=120000 * 6), tensorboard_log_dir=config.tensorboard_log_dir))

    llm = get_default_fm(get_response=get_response)
    llm.retry_times = 30

    mobile_api = MobileAPI(config=MobileAPIConfig(
        locator='ui_tars',
        device_serial_id=config.serial,
        llm=llm,
        locator_config=None
    ))
    def get_screenshot_dict(agent_prog_context: AgentProgContext):
        return agent_prog_context.workflow_context.global_vars.get(SCREENSHOT_DICT, {})
    def get_screenshot_func(image_dir: Path, text_mode: bool=False, text_augment: bool=False):
        img_count = 0
        if image_dir.exists():
            shutil.rmtree(image_dir)
        def get_screenshot():
            nonlocal img_count
            image_filename = Path(f"screenshot_{img_count}.png")
            image_filepath = image_dir / image_filename
            image_filepath.parent.mkdir(parents=True, exist_ok=True)
            screenshot: Image.Image = mobile_api.take_screenshot()
            resized_screenshot = screenshot.resize((round(screenshot.width / 4), round(screenshot.height / 4)))
            resized_screenshot.save(image_filepath)
            img_count += 1
            if not text_mode:
                image_id = image_filepath.resolve()
                return f'{IMAGE_START}{image_id}{IMAGE_END}' + ("\nText Description: " + get_text_description(resized_screenshot, llm) if text_augment else '')
            else:
                return get_text_description(resized_screenshot)
        return get_screenshot
    
    task_description = config.task_description
    workflow_script = Path(config.workflow_path).read_text() # anything begins from nothing.

    inject_global_vars = {
        "llm": llm,
        "mobile": mobile_api,
        SCREENSHOT_FUNC: get_screenshot_func(image_dir=Path(config.image_dir), text_augment=False, text_mode=False),
        SCREENSHOT_DICT: {}
    }
    def workflow_callback(global_vars, local_vars):
        meta_info_dir = Path(config.meta_info_dir)
        if config.meta_info_dir:
            meta_info_dir.mkdir(parents=True, exist_ok=True)
            (meta_info_dir / "completion_statistics.json").write_text(json.dumps(dump_completion_statistics_dict()))

        exec(f"""
current_screenshot = {SCREENSHOT_FUNC}()
if current_screenshot.startswith('{IMAGE_START}') and current_screenshot.endswith('{IMAGE_END}'):
    {HIDDEN_VARS_PREFIX}_image_id = current_screenshot.removeprefix('{IMAGE_START}').removesuffix('{IMAGE_END}')
    {SCREENSHOT_DICT}[{HIDDEN_VARS_PREFIX}_image_id]={HIDDEN_VARS_PREFIX}_image_id
""", global_vars, local_vars)
    from agentprog.plan.code_exec.workflow.workflow_prompts.update_belief_state_prompt import update_belief_state_prompt_mobile
    match RequestMode[config.request_mode]:
        case RequestMode.api:
            # api mode
            from agentprog.plan.code_exec.workflow.workflow_prompts.workflow_prompt_set import CoTMobileWorkflowPromptSet, CoTMobileWorkflowPromptSetWithBeliefState

            workflow_prompt_set_cls = CoTMobileWorkflowPromptSetWithBeliefState if config.use_belief_state else CoTMobileWorkflowPromptSet
            return run_workflow(config, get_response, task_description, workflow_script, workflow_prompt_set=workflow_prompt_set_cls(get_images=get_screenshot_dict, update_belief_state=update_belief_state_prompt_mobile, get_framework_prompt=(lambda : get_mobile_prompt(get_date_on_mobile_phone(config.serial)))), inject_global_vars=inject_global_vars, workflow_callback=workflow_callback)


def _agentprog_run_ai_phone(config: AgentProgConfig):
    from agentprog.all_utils.mobile_utils import get_text_description, MobileAPI, MobileAPIConfig
    from agentprog.all_utils.fm import get_default_fm
    SCREENSHOT_FUNC = f"{HIDDEN_VARS_PREFIX}_get_screenshot"
    SCREENSHOT_DICT = f"{HIDDEN_VARS_PREFIX}_screenshot_dict"
    get_response = init_get_gemini_response(init_response_args=InitResponseArgs(model='vertex_ai/gemini-2.5-pro', record_completion_statistics=True, token_budget=TokenStatistics(prompt_tokens=550000 * 6, completion_tokens=120000 * 6), tensorboard_log_dir=config.tensorboard_log_dir))

    llm = get_default_fm(get_response=get_response)
    llm.retry_times = 30

    mobile_api = MobileAPI(config=MobileAPIConfig(
        locator='ui_tars',
        device_serial_id=config.serial,
        llm=llm,
        locator_config=None
    ))
    def get_screenshot_dict(agent_prog_context: AgentProgContext):
        return agent_prog_context.workflow_context.global_vars.get(SCREENSHOT_DICT, {})
    def get_screenshot_func(image_dir: Path, text_mode: bool=False, text_augment: bool=False):
        img_count = 0
        if image_dir.exists():
            shutil.rmtree(image_dir)
        def get_screenshot():
            nonlocal img_count
            image_filename = Path(f"screenshot_{img_count}.png")
            image_filepath = image_dir / image_filename
            image_filepath.parent.mkdir(parents=True, exist_ok=True)
            screenshot: Image.Image = mobile_api.take_screenshot()
            resized_screenshot = screenshot.resize((round(screenshot.width / 4), round(screenshot.height / 4)))
            resized_screenshot.save(image_filepath)
            img_count += 1
            if not text_mode:
                image_id = image_filepath.resolve()
                return f'{IMAGE_START}{image_id}{IMAGE_END}' + ("\nText Description: " + get_text_description(resized_screenshot, llm) if text_augment else '')
            else:
                return get_text_description(resized_screenshot)
        return get_screenshot
    
    task_description = config.task_description
    workflow_script = Path(config.workflow_path).read_text() # anything begins from nothing.

    inject_global_vars = {
        "llm": llm,
        "mobile": mobile_api,
        SCREENSHOT_FUNC: get_screenshot_func(image_dir=Path(config.image_dir), text_augment=True, text_mode=False),
        SCREENSHOT_DICT: {}
    }
    def workflow_callback(global_vars, local_vars):
        meta_info_dir = Path(config.meta_info_dir)
        if config.meta_info_dir:
            meta_info_dir.mkdir(parents=True, exist_ok=True)
            (meta_info_dir / "completion_statistics.json").write_text(json.dumps(dump_completion_statistics_dict()))

        exec(f"""
current_screenshot = {SCREENSHOT_FUNC}()
if current_screenshot.startswith('{IMAGE_START}') and current_screenshot.endswith('{IMAGE_END}'):
    {HIDDEN_VARS_PREFIX}_image_id = current_screenshot.removeprefix('{IMAGE_START}').removesuffix('{IMAGE_END}')
    {SCREENSHOT_DICT}[{HIDDEN_VARS_PREFIX}_image_id]={HIDDEN_VARS_PREFIX}_image_id
""", global_vars, local_vars)
    from agentprog.plan.code_exec.workflow.workflow_prompts.update_belief_state_prompt import update_belief_state_prompt_mobile
    match RequestMode[config.request_mode]:
        case RequestMode.api:
            # api mode
            from agentprog.plan.code_exec.workflow.workflow_prompts.workflow_prompt_set import CoTMobileWorkflowPromptSet, CoTMobileWorkflowPromptSetWithBeliefState

            workflow_prompt_set_cls = CoTMobileWorkflowPromptSetWithBeliefState if config.use_belief_state else CoTMobileWorkflowPromptSet
            return run_workflow(config, get_response, task_description, workflow_script, workflow_prompt_set=workflow_prompt_set_cls(get_images=get_screenshot_dict, update_belief_state=update_belief_state_prompt_mobile, get_framework_prompt=(lambda : get_mobile_prompt(get_date_on_mobile_phone(config.serial)))), inject_global_vars=inject_global_vars, workflow_callback=workflow_callback, cache_mode=config.cache_mode, use_belief_state=config.use_belief_state)

def agentprog_run_core(config: AgentProgConfig):
    _agentprog_run_router = {
        ToolSet.mobile: _agentprog_run_mobile,
        ToolSet.ai_phone: _agentprog_run_ai_phone,
    }

    workflow_result = _agentprog_run_router[ToolSet[config.tool_set]](config)
    
    return workflow_result

def agentprog_run_cli(args: list[str]=None):
    arg_parser = ArgumentParser()
    arg_parser.add_argument("workflow_path", type=str, help="workflow path to be executed")
    arg_parser.add_argument("--task_description", default=None, type=str, required=False, help="Total description of the task")
    add_common_args(arg_parser)

    cli_args = arg_parser.parse_args(args)

    file_arg_list = []
    with open(cli_args.workflow_path, "r") as f:
        first_line = f.readline().strip()
        if first_line.startswith(CLI_LEADER):
            file_arg_list = shlex.split(first_line.removeprefix(CLI_LEADER))
    file_args = arg_parser.parse_args([cli_args.workflow_path] + file_arg_list)
    
    full_args = Namespace(**vars(cli_args))
    # if nothing given, set to default
    for default_key in AgentProgConfig.get_field_names():
        for try_value in map(lambda s: s.get(default_key, None), (vars(cli_args), vars(file_args), AgentProgConfig.get_field_default_value())):
            if try_value is not None:
                vars(full_args)[default_key] = try_value
                break

    if full_args.logging_path is not None:
        from agentprog.all_utils import log_utils
        log_utils.global_log_config.logging_path = full_args.logging_path

    config = AgentProgConfig(**vars(full_args))

    start_time = time.perf_counter()

    workflow_result = agentprog_run_core(config)

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Task Program Running Time: {execution_time:.4f} seconds")
    # write log
    meta_info_dir = Path(full_args.meta_info_dir)
    meta_info_dir.mkdir(parents=True, exist_ok=True)
    (meta_info_dir / "completion_statistics.json").write_text(json.dumps(dump_completion_statistics_dict()))
    return workflow_result
    
if __name__ == "__main__":
    pass