# 返回值要么为一个 Prompt，要么为一个 Cached Answer。
from dataclasses import field, dataclass
from enum import Enum, auto
import traceback
from agentprog.all_utils.fm import FoundationModel
from agentprog.all_utils.mobile_utils import MobileAPI
from agentprog.plan.belief.belief_state import BeliefState
from agentprog.plan.workflow_utils import WorkflowContext, LLMQueryMode, WorkflowProgramCounterOperation, collect_variables_list, summarize_variables
import os

from dotenv import load_dotenv; load_dotenv(override=True)

class RequestMode(Enum):
    api = auto()

class ToolSet(Enum):
    mobile = auto()

ignore_class_attr = (FoundationModel, MobileAPI)
hidden_class_attr = (FoundationModel, MobileAPI)

@dataclass
class PromptResult:
    prompt_str: str = ""

@dataclass 
class CachedResult:
    cached_answer: str = ""

@dataclass
class AgentProgContext: # 存储从 workflow_context 中解析的临时信息，只是对 WorkflowContext 的某种包装罢了！
    workflow_context: WorkflowContext = None
    task_description: str = ''
    workflow_context_str: str = ''
    data_and_variables: str = ''
    python_context: str = ''
    current_line: str = ''
    llm_query_mode: LLMQueryMode = None
    preparations: str = '' # 执行 workflow 之前的准备工作
    belief_state: BeliefState = field(default_factory=lambda: BeliefState('', ''))

    def to_json(self):
        return {
            "workflow_context": self.workflow_context.get_workflow_tree(include_children=False),
            "task_description": self.task_description,
            "workflow_context_str": self.workflow_context_str,
            "data_and_variables": self.data_and_variables,
            "python_context": self.python_context,
            "current_line": self.current_line,
            "llm_query_mode": str(self.llm_query_mode.name)
        }

    @classmethod
    def from_json(cls, json_dict: dict):
        return cls(
            workflow_context=WorkflowContext.from_json(json_dict['workflow_context']),
            task_description=json_dict.get('task_description'),
            workflow_context_str=json_dict.get('workflow_context_str'),
            data_and_variables=json_dict.get('data_and_variables'),
            python_context=json_dict.get('python_context'),
            current_line=json_dict.get('current_line'),
            llm_query_mode=LLMQueryMode[json_dict.get('llm_query_mode').removeprefix(f"{LLMQueryMode.__name__}.")],
            preparations=json_dict.get('preparations'),
        )


@dataclass
class AgentProgRawResponse: # 解析前的 LLM Response
    content: str
    mode: LLMQueryMode

@dataclass
class AgentProgParsedResponse: # 解析后的结构化数据
    script: str = None
    wpc_op: WorkflowProgramCounterOperation = None
    thought: str = ''
    observation: str = ''
    belief_state: str = ''
    plan: str = ''
    mode: LLMQueryMode | None = None

def _split_thought_action(raw_string: str):
    split_index = raw_string.index("--- Action ---")
    return raw_string[:split_index].strip().removeprefix("--- Thought ---"), raw_string[split_index:].strip().removeprefix("--- Action ---")

def _split_fields(raw_string: str, field_list: list[str]):
    split_by_text = lambda s, t: (lambda i : (s[:i], s[i + len(t):]))(s.index(t))
    remaining = raw_string
    result_dict = {}
    for field_idx, (field_name, field_prefix) in list(enumerate(field_list))[::-1]:
        if field_prefix in remaining:
            remaining, field_str = split_by_text(remaining, "" if field_idx == 0 else '\n' + field_prefix.strip())
            result_dict[field_name] = field_str.strip().removeprefix(field_prefix) if field_idx == 0 else field_str.strip()
        else:
            result_dict[field_name] = ''
    return result_dict

def _parse_thought_script(raw_response: AgentProgRawResponse):
    '''
    回答中包含 action & thought
    '''
    thought, action = _split_thought_action(raw_response.content)
    script = action.replace("```python", "").replace("```", "").strip()
    match raw_response.mode:
        case LLMQueryMode.WorkflowStatusUpdate:
            for test_wpc_op_name in WorkflowProgramCounterOperation._member_names_:
                if test_wpc_op_name.lower() in script.lower():
                    return AgentProgParsedResponse(
                            script=script,
                            thought=thought,
                            wpc_op=WorkflowProgramCounterOperation[test_wpc_op_name],
                            mode=raw_response.mode
                        )
            raise ValueError(f"Invalid Workflow Program Counter Operation `{script}`, you can only choose from: {WorkflowProgramCounterOperation._member_names_}!")
        
        case LLMQueryMode.CodeGeneration:
            return AgentProgParsedResponse(
                script=script,
                thought=thought,
                mode=raw_response.mode
            )

def _parse_script_only(raw_response: AgentProgRawResponse):
    '''
    如果回答中只有 script 就这样匹配
    '''
    content = raw_response.content
    script = content.replace("```python", "").replace("```", "").strip()
    match raw_response.mode:
        case LLMQueryMode.WorkflowStatusUpdate:
            for test_wpc_op_name in WorkflowProgramCounterOperation._member_names_:
                if test_wpc_op_name.lower() in script.lower():
                    return AgentProgParsedResponse(
                            script=script,
                            wpc_op=WorkflowProgramCounterOperation[test_wpc_op_name],
                            mode=raw_response.mode
                        )
            raise ValueError(f"Invalid Workflow Program Counter Operation `{script}`, you can only choose from: {WorkflowProgramCounterOperation._member_names_}!")
        
        case LLMQueryMode.CodeGeneration:
            return AgentProgParsedResponse(
                script=script,
                mode=raw_response.mode
            )

def _parse_script_w_belief_state(raw_response: AgentProgRawResponse):
    '''
    回答中包含 action & thought
    '''
    field_list = [("observation", "--- Observation ---"), ("thought", "--- Thought ---"), ("belief_state", "Updated Belief --- State ---"), ("judgement", "--- Judgement ---"), ("plan", "--- Plan ---"), ("action", "--- Action ---")]
    parsed_fields = _split_fields(raw_response.content, field_list)
    script = parsed_fields['action'].replace("```python", "").replace("```", "").strip()
    match raw_response.mode:
        case LLMQueryMode.WorkflowStatusUpdate:
            for test_wpc_op_name in WorkflowProgramCounterOperation._member_names_:
                if test_wpc_op_name.lower() in script.lower():
                    return AgentProgParsedResponse(
                            script=script,
                            thought=parsed_fields['thought'],
                            wpc_op=WorkflowProgramCounterOperation[test_wpc_op_name],
                            observation=parsed_fields['observation'],
                            belief_state=parsed_fields['belief_state'],
                            plan=parsed_fields['plan'],
                            mode=raw_response.mode
                        )
            raise ValueError(f"Invalid Workflow Program Counter Operation `{script}`, you can only choose from: {WorkflowProgramCounterOperation._member_names_}!")
        
        case LLMQueryMode.CodeGeneration:
            return AgentProgParsedResponse(
                script=script,
                thought=parsed_fields['thought'],
                observation=parsed_fields['observation'],
                belief_state=parsed_fields['belief_state'],
                plan=parsed_fields['plan'],
                mode=raw_response.mode
            )

def filter_variables_list(global_vars, local_vars):
    if global_vars is local_vars:
        # print global only
        return collect_variables_list(global_vars, ignore_class_attr=ignore_class_attr, hidden_class_attr=hidden_class_attr)
    else:
        return collect_variables_list(global_vars, ignore_class_attr=ignore_class_attr, hidden_class_attr=hidden_class_attr) + collect_variables_list(local_vars, ignore_class_attr=(), hidden_class_attr=hidden_class_attr)

def print_variables(global_vars, local_vars):
    if global_vars is local_vars:
        # print global only
        return f"""
global variables:
{summarize_variables(global_vars, ignore_class_attr=(FoundationModel,), hidden_class_attr=())}
"""
    else:
        return f"""
global variables:
{summarize_variables(global_vars, ignore_class_attr=(FoundationModel,), hidden_class_attr=())}

local variables:
{summarize_variables(local_vars, ignore_class_attr=(), hidden_class_attr=())}
"""
