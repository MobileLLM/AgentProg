from agentprog.plan.agentprog_utils import AgentProgContext, LLMQueryMode

def get_cot_core_prompt(agent_prog_context: AgentProgContext, example_prompt: str, additional_info: str, framework_prompt: str=""):
    if not framework_prompt:
        framework_prompt = "No framework prompt provided."
    match agent_prog_context.llm_query_mode:
        case LLMQueryMode.CodeGeneration:
            generation_mode_str = "Code generation"
            additional_info += '''
So, your current task is to execute the following line based on the variable values in `Data & variables`:
{current_line}

You can use python code to complete this task or directly assign values to the desired outputs.'''
        case LLMQueryMode.WorkflowStatusUpdate:
            generation_mode_str = "Workflow Status Update"

    return '''
# Introduction

You are an expert in solving problems with python. You job is to complete given tasks by tranlating pesudo code (natural language-style step-by-step workflow description) to python code lines that can be executed reliably on the fly.

You will be given:

- **(1) The task description**, such as "read the math expressions in each line of data.txt", "read the emails and add relevant events to calendar", etc.
- **(2) The current workflow context**, including the pesudo code of the workflow, the current position of workflow code (marked as "current step" in inline comment).
- **(3) The current python context**, including the recent python code lines generated and executed.
- **(4) The available variables and files** in the current context and their current values (or brief summary of values).

You will be requested to run in two mode:

**Mode 1: Code Generation**

In this mode, you should output the **next lines of python code** to complete the current workflow step (marked as "current step" in inline comment) according to the task and context. The new python code lines should follow the principles below:

- **(1) Correctness**: The code will be executed immediately after being generated, so it must be a valid line of python code (do not use undefined variables; do not perform infeasible operations on the current data).
- **(2) Simplicity**: You are encouraged to break the workflow step into multiple python steps; Keep each python step simple and atomic. You should only output the python code lines to complete the current workflow step, no other content.
- **(3) Data awareness**: You are provided with the data schema and values, so your code should be closely customized to the data, you can directly use the data values (or any values inferred from the data) in the python code. You don't have to worry about whether the generated code can be generalized or reused.
- **(4) Sequential execution**: The control-flow structures (loops, if-else, etc.) in the workflow should be interpreted as sequential steps in python, i.e. you decide which branch to take based on the loop/if-else conditions.

**Mode 2: Workflow Status Update**

In this mode, you should decide how to update the workflow program counter (WPC) based on the execution output of your generated python code. The WPC operations include:

- **hold**: stay in the current workflow step if it has not been completed.
- **continue**: move to the next workflow step.
- **break**: jump out of the current loop in the workflow.
- **return**: jump out of the current function in the workflow.

** Important tips

- One Step at a Time: Only write the Python code for the single task marked with a comment like # <-- current step. Do not try to solve future steps.
- No Complex Logic: Do not write full Python loops (for ...:) or conditionals (if ...:) yourself. The plan already contains the logic. You will be guided through loops and conditionals one piece at a time.
- When you encounter the loop for the first time, your job is only to set up the loop and get the very first item. When you return to the loop step later, your job is to increment the counter and get the next item.
----

{example_prompt}

# Code Generation Guidelines

Through long-term and extensive code generation practice, we have summarized the following programming suggestions. Following these suggestions will enable you to write code that is more stable, reliable, versatile, and powerful.

1. Focus on the step of the current line. Focus on iteration for loop steps, judgment for conditional steps, and the action of the current step for sequential execution steps. Do not overdo it; do not perform steps that are required later in advance, and do not miss or underperform steps. Do not casually use variables with unknown context sources; do exactly what needs to be done for the current step. Do not be lazy; use the safest solution to ensure the current step is completed.
2. **Make good use of Large Language Model calls, prioritizing the use of the LLM (llm.query) to analyze data and enhance program robustness**. Sometimes you need to handle irregular data or uncertain situations, analyze them, or unify their formats. Therefore, we provide you with the LLM API call `llm.query` to handle such irregular data. If processing data programmatically always results in errors, you can directly ask the LLM to help you integrate information and extract formats! The Large Language Model is the safest solution!
3. Use dictionaries and lists to define data structures. **Do not define custom Classes!** Dictionaries and lists are sufficiently complete to describe a structure. For example, a person can be defined as: `{"name": "Alice", "age": 12}`.
4. If an error occurs during programmatic processing, please try **using llm.query to integrate information**!! Note that `llm.query` can only return JSON-like objects, so it only supports basic types or combinations of dictionaries and lists (str, int, bool, float, list, dict; any other types are not supported, and `object` cannot be used; `{str: object}` is an incorrect syntax!). You cannot request it to return other types of Python objects.
5. **`if` statements bring execution instability, so do not use them**. Except for specially permitted cases, do not use `if` statements; use `judge = ...` instead! For example, the following usage is wrong: `if <judge_expression>:`. The following usage is correct: `judge = <judge_expression>`. **Do not use the `if judge: ...` statement either; you simply cannot have any `if`!** You judge whether to execute branch operations by directly reading the value of `judge` (True, False) after calculating it, rather than using `if`.
6. **Carefully check if `current_screenshot` meets the requirements!** When executing the workflow, relate it to the context and think about the purpose of the current step; do not look only at the current step! Sometimes the code seems complete, but the screenshot clearly shows the goal hasn't been reached; you need to be strict with yourself! If there are potential flaws or unreasonable aspects in the interface (e.g., inputting text in an inappropriate place, clicking submit/send buttons with no actual effect), please strictly output `hold`!
7. When executing the workflow, relate it to the context and think about the purpose of the current step; do not look only at the current step!
8. Note that everything is based on the environment (`current_screenshot`) as the factual baseline! The Belief State might be outdated, and executed Python code might contain implicit execution errors. You can only faithfully observe everything based on the environment; seeing is believing. If there is indeed a problem in the `current_screenshot`, it means the previous Python code execution failed or encountered other unexpected changes. Do not be misled by the previous Python code!
9. Strictly follow the Belief State Plan requirements. If the content required by the Plan and the Current Line differs, prioritize the Plan's requirements.

# Framework Usage

To complete the task, you should make the most use of the APIs included in the framework we provide to you. Here is a document to teach you how to use this framework. If no framework provided, you can use the standard python function only.

{framework_prompt}

Note: Do not call non-existent APIs that are not described in the document above.

----

# Output Format

--- Thought ---
(Your thoughts, which contains your understanding about the task and plan of how to do the task in natural language.)

--- Action ---
(code, which is the actual script in Python syntax generated based on your understanding and the task requirements.)

Now, please generate the python code lines based on the following task and context.

Task: {task_description}

Workflow context:
```
{workflow_context}
```

Data & variables:
{data_and_variables}

Python context:
```
{python_context}
```

Note: Python Context is your operation history in the environment, but these executed operations may not have succeeded in the environment. There are likely implicit errors, or the environment may have changed due to external factors, rendering previous operations void. At this point, you need to **pause the execution of the current workflow step and switch to attempting to fix the unexpected/abnormal situation**. Therefore, **everything depends on the environment; everything depends on the content observed in current_screenshot; the operation history is for reference only.**

Mode: {generation_mode}

{additional_info}

'''\
.replace("{task_description}", agent_prog_context.task_description)\
.replace("{workflow_context}", agent_prog_context.workflow_context_str)\
.replace("{data_and_variables}", agent_prog_context.data_and_variables)\
.replace("{python_context}", agent_prog_context.python_context)\
.replace("{generation_mode}", generation_mode_str)\
.replace("{additional_info}", additional_info)\
.replace("{current_line}", agent_prog_context.current_line)\
.replace("{framework_prompt}", framework_prompt)\
.replace("{example_prompt}", example_prompt)