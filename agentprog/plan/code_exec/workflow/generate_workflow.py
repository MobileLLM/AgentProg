from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any
import time
import math
from agentprog.all_utils.general_utils import Messages, init_get_openai_response, init_get_parsed_response, make_user, init_get_claude_response, init_get_gemini_response, InitResponseArgs, TokenStatistics
from agentprog.plan.code_exec.workflow.workflow_prompts.generate_workflow_prompts.prompt_mobile import get_script_mobile
from agentprog.plan.workflow_utils import compile_workflow

token_budget = TokenStatistics(prompt_tokens=60000, completion_tokens=math.inf)
get_response = init_get_gemini_response(init_response_args=InitResponseArgs(model='vertex_ai/gemini-2.5-pro', record_completion_statistics=True, token_budget=token_budget))
from agentprog.all_utils import log_utils
logger = log_utils.get_logger(__name__)

@dataclass
class GenerateWorkflowArgs:
    task_description: str
    output_path: str
    get_workflow_response: Any=None
    logging_path: str = None

def _parse_thought_workflow(response: str):
    split_index = response.index("--- Workflow ---")
    return response[:split_index].strip().removeprefix("--- Thought ---"), response[split_index:].strip().removeprefix("--- Workflow ---").replace("```agentprog", "").replace("```", "")

def _check_workflow_valid(workflow: str):
    compiled_workflow, _, _ = compile_workflow(workflow)
    code = compile(compiled_workflow, "<test>", "exec")
    return code

get_parsed_response = init_get_parsed_response(get_response, lambda r: (lambda t, w: (lambda c: (t, w))(_check_workflow_valid(w)))(*_parse_thought_workflow(r)), try_times=6)

def generate_script(task_description, get_prompt, get_response, response_parser, output_path="llm_generated.ap"):
    context_logger = logger.bind(task_description=task_description, get_response=get_response.__name__, get_prompt=get_prompt.__name__, output_path=output_path)

    context_logger.info("start timing...")
    start_time = time.perf_counter()
    
    prompt = get_prompt(task_description)
    context_logger.info("trying to request")

    (thought, code), res = get_parsed_response(Messages([make_user(prompt)]).serialize())
    context_logger.info("request finished successfully")
    

    end_time = time.perf_counter()
    context_logger.info("end timing...")

    execution_time = end_time - start_time
    logger.info("=" * 20)
    logger.info(res)
    logger.info("=" * 20)
    logger.info("=" * 20)
    logger.info(thought)
    logger.info("=" * 20)
    logger.info(code)
    logger.info("=" * 20)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"# {execution_time:.4f} seconds" + "\n" + code)
    context_logger.info("write code successfully")

    logger.info(f"generation time: {execution_time:.4f} seconds")

def query_llm(get_response, prompt, output_path):
    start_time = time.perf_counter()
    res = get_response([make_user(prompt).serialize()])
    end_time = time.perf_counter()
    
    execution_time = end_time - start_time
    logger.info(f"generation time: {execution_time:.4f} seconds")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"# {execution_time:.4f} seconds" + "\n" + str(res))

def handle_generate_workflow_args(generate_workflow_args: GenerateWorkflowArgs):
    if generate_workflow_args.logging_path is not None:
        log_utils.global_log_config.logging_path = generate_workflow_args.logging_path

def generate_workflow(generate_workflow_args: GenerateWorkflowArgs):
    handle_generate_workflow_args(generate_workflow_args)
    global get_response
    res = generate_script(generate_workflow_args.task_description, get_prompt=get_script_mobile, get_response=generate_workflow_args.get_workflow_response or get_response, response_parser=_parse_thought_workflow, output_path=generate_workflow_args.output_path)
    return res
