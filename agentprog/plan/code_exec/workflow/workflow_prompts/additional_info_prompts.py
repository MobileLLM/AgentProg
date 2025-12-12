from typing import Any, Callable
from agentprog.plan.agentprog_utils import AgentProgContext
from agentprog.plan.workflow_utils import ERROR_INFO_NAME, WorkflowNodeType, grab_similar_nodes

def get_additional_info_for_code_generation_mobile(agent_prog_context: AgentProgContext, similar_workflows=None):
    workflow_context = agent_prog_context.workflow_context
    if similar_workflows is None:
        similar_workflows = grab_similar_nodes(workflow_context)
    additional_info = ""

    additional_info += f"Observe whether the `{ERROR_INFO_NAME}` variable in the Python Context indicates an error. If an error is reported, it means the code you executed in the Python Context was not successful. You need to learn from previous experiences and lessons, adjust your strategy, and try to solve the problem again. Do not repeat the same mistakes.\n"
    if workflow_context is not None and workflow_context.workflow_reflection:
        additional_info += f"Note: According to the review, the code executed so far has not yet achieved the goal of the current step (the current step is not finished, or an error was thrown during execution). The review feedback is as follows: {workflow_context.workflow_reflection}\nPlease generate new code based on the review feedback to make progress towards the goal.\n(Some hints: 1. If the desired item is not in the interface, perhaps it just isn't displayed. Try exploring more. If there is a search function, try using it to find what you want. If not, try swiping down/right for a distance, or look for expand/next page buttons and click them. 2. Sometimes even if operations like click or input do not report errors, it is possible the wrong place was clicked. Therefore, you need to analyze the interface itself carefully. If it doesn't meet expectations, raise the issue. If multiple attempts fail, try exiting and starting over. 3. Check if mobile.check was performed and if its return value is True.)"

    # Currently we only call the most recent one
    if similar_workflows:
        ref_workflow = similar_workflows[0]
        additional_info += f"You have executed a similar step before. The code you provided then was as follows:\n```\n{ref_workflow.script}\n```\nYou can selectively reference the previous code to complete the current task. If it is ineffective, consider other strategies.\n"

    if workflow_context.workflow_node_type == WorkflowNodeType.LOOP_FOR:
        additional_info += f"This is your {workflow_context.duplicate_context_id + 1}'th time to reach this line of code.\n"
        additional_info += "Current line is the beginning of a loop body, you should take a value from a iterable object. (If you enter this loop first time, you should create the iterable object first.) "
        if workflow_context.duplicate_context_id > 0:
            # Loop, get the first cache_code and the last cache_code
            cache_node_list = workflow_context.cache_node_list
            first_node = cache_node_list[0]
            last_node = cache_node_list[-1]
            additional_info += f"When you reach this line for the first time, you generated these code: \n```{first_node.script}```\n"
            if last_node is not first_node:
                additional_info += f"When you reach this line for the most recent time, you generated these code: \n```{last_node.script}```\n"

    if workflow_context.workflow_node_type == WorkflowNodeType.CONDITION:
        additional_info += "Currently at a conditional statement node. Please generate code for the judgment condition based on the conditional statement, such as judge = .... If the condition judgment is difficult, you can call llm.query to let the large model help you judge."

    if workflow_context.workflow_node_type == WorkflowNodeType.CONDITION_INLINE:
        additional_info += """Currently at an inline conditional statement node. Please generate code for the judgment condition based on the conditional statement, such as judge = ..., and then generate the 'operations that should be executed if the condition is met'. Like this:
```
judge = <condition expresion>
if judge:
    <expression> # Operations to execute if condition is met
```
If the condition judgment is difficult, you can call llm.query to let the large model help you judge.        
"""
    return additional_info

def get_additional_info_for_workflow_status_update_mobile(agent_prog_context: AgentProgContext, similar_workflows=None):
    workflow_context = agent_prog_context.workflow_context

    additional_info = ""
    additional_info += "It is now Workflow Status Update mode. You need to and can only select one of the following WPC operations as Action: hold, continue, break, return. First, you need to observe whether `error_info` exists. If there is error information, you **must** point out the problem in time, propose modification suggestions, and output `hold`. Secondly, please judge based on the information in Context and Variable whether the code generated last time has strictly completed the step indicated by `current step`. If not yet satisfied, please point out the problem, make suggestions, and output `hold`. For mobile operation tasks, usually multiple steps are required to complete. Therefore, please carefully check whether the goal of `current step` is confirmed to be fully completed, and whether the belief state has raised noteworthy issues (such as unexpected situations outside the plan that need correction). If not completed, output `hold`."

    if workflow_context.workflow_node_type == WorkflowNodeType.LOOP_FOR:
        additional_info += "Current line is the beginning of a for-loop body, you should check whether a value is taken from a iterable object successfully. If so, you should output `continue`. Else, if an StopIteration error raised, you should output `break`. If an unexpected error raised or there is problem in the plan of belief state, you should output `hold`. Output `continue` if and only if the `current step` is confirmed complete, there is no error information, and the belief state considers that the current status does not require exception handling."
    
    elif workflow_context.workflow_node_type == WorkflowNodeType.LOOP_WHILE:
        additional_info += "Current line is the beginning of a while-loop body. Please check whether the conditional statement currently being executed is true based on the information in Context and Variable and the execution result of the code. If the `judge` variable value already exists, you **must** directly refer to the `judge` variable being True/False as the basis; `judge` being True means it is true, and `judge` being False means it is false. If `judge` is true, you should output `continue`; if `judge` is false, you should output `break`. If an unexpected error raised, you should output `hold`."

    elif workflow_context.workflow_node_type == WorkflowNodeType.CONDITION or workflow_context.workflow_node_type == WorkflowNodeType.CONDITION_INLINE:
        additional_info += "The current node is a condition judgment node. Please check whether the conditional statement currently being executed is true based on the information in Context and Variable and the execution result of the code. If the `judge` variable value already exists, you **must** directly refer to the `judge` variable being True/False as the basis; `judge` being True means it is true, and `judge` being False means it is false. If true, you should output `continue`; otherwise, you should output `break`."

    elif workflow_context.workflow_node_type == WorkflowNodeType.SEQUENTIAL or workflow_context.workflow_node_type == WorkflowNodeType.RETURN:
        additional_info += "The current node is a sequential execution node. Output `continue` if and only if the `current step` is confirmed complete, there is no error information, and the belief state considers that the current status does not require exception handling."
    
    additional_info += "Emphasize again: You **need to and can only** select one of the following WPC operations as Action: hold, continue, break, return."

    return additional_info