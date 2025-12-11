from __future__ import annotations
from argparse import ArgumentParser
from agentprog.all_utils.general_utils import InitResponseArgs, init_get_gemini_response
from agentprog.plan.code_exec.workflow.config.core_config import AgentProgConfig
from agentprog.plan.code_exec.workflow.config.utils import add_common_args
from agentprog.plan.code_exec.workflow.generate_workflow import GenerateWorkflowArgs, generate_workflow
from agentprog.plan.code_exec.workflow.workflow_query_llm import agentprog_run_core
from datetime import datetime

def agentprog_pipeline_core(config: AgentProgConfig):
    # generate workflow
    task_description = config.task_description
    workflow_path = config.workflow_path
    tensorboard_log_dir = config.tensorboard_log_dir
    logging_path = config.logging_path

    get_generate_response = init_get_gemini_response(init_response_args=InitResponseArgs(model='vertex_ai/gemini-2.5-pro', record_completion_statistics=True, tensorboard_log_dir=tensorboard_log_dir))
    generate_workflow_args = GenerateWorkflowArgs(task_description, workflow_path, get_generate_response, logging_path)
    generate_workflow(generate_workflow_args)

    workflow_result = agentprog_run_core(config)
    return workflow_result

def agentprog_pipeline_cli(args: list[str]=None):
    arg_parser = ArgumentParser()
    arg_parser.add_argument("task_description", type=str, help="")
    arg_parser.add_argument("--workflow_path", type=str, required=False, default=f"outputs/{datetime.now().strftime('%Y%m%d%H%M%S')}.ap", help="workflow path to be generated and executed")
    add_common_args(arg_parser)
    
    cli_args = arg_parser.parse_args(args)
    full_args = cli_args

    for default_key in AgentProgConfig.get_field_names():
        for try_value in map(lambda s: s.get(default_key, None), (vars(cli_args), AgentProgConfig.get_field_default_value())):
            if try_value is not None:
                vars(full_args)[default_key] = try_value
                break
        
    if full_args.logging_path is not None:
        from agentprog.all_utils import log_utils
        log_utils.global_log_config.logging_path = full_args.logging_path

    # run task program
    config = AgentProgConfig(**vars(full_args))
    workflow_result = agentprog_pipeline_core(config)

    return workflow_result