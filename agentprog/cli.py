import sys

def agentprog_cli():
    argv = sys.argv[1:]
    if argv:
        command = argv[0]
        if command == 'run':
            from agentprog.plan.code_exec.workflow.workflow_query_llm import agentprog_run_cli
            agentprog_run_cli(argv[1:]) # 把 run 也拆分掉。
        else:
            from agentprog.plan.code_exec.workflow.pipeline import agentprog_pipeline_cli
            agentprog_pipeline_cli(argv)
    else:
        from agentprog.plan.tui.tui_app import agentprog_interaction_cli
        agentprog_interaction_cli()