from __future__ import annotations
from enum import auto, Enum
import traceback
from typing import Callable, Dict, List
from agentprog.all_utils import log_utils
from agentprog.plan.belief.belief_state import BeliefState
from agentprog.plan.code_exec.workflow.config.core_config import AgentProgConfig
from agentprog.plan.code_exec.workflow.workflow_prompts.workflow_prompt_set import StandardWorkflowPromptSet, WorkflowPromptSetOptional
from agentprog.all_utils.debug import need_breakpoint
from agentprog.all_utils.general_utils import Messages, Message, Prompt, init_get_parsed_response
from agentprog.plan.workflow_utils import EXEC_RESULT_COMMENT, WORKFLOW_NAME, SNIPE_NAME, WORKFLOW_STEP_COMMENT, LLMQueryMode, WorkflowContext, WorkflowNodeType, WorkflowProgramCounterOperation, WorkflowResult, WorkflowRoot, WorkflowSystem, grab_similar_nodes
from agentprog.plan.agentprog_utils import CachedResult, AgentProgContext, AgentProgParsedResponse, AgentProgRawResponse, PromptResult, print_variables, show_dashboard

logger = log_utils.get_logger(__name__)

class AgentProgEvent(Enum):
    before_code_generation = auto()    

class WorkflowEvent(Enum):
    before_workflow_execution = auto()
    after_workflow_execution = auto()

class AgentProgCore:
    def __init__(self, config: AgentProgConfig, get_response: Callable[[List[Dict[str, str]]], str], workflow_prompt_set: StandardWorkflowPromptSet, get_fix_response=None):
        self.config = config
        self.get_response = get_response
        self.workflow_prompt_set = workflow_prompt_set
        self.belief_state = BeliefState("No Belief State Now.", "")
        self.cache_mode = config.cache_mode
        self.use_belief_state = config.use_belief_state
        self.event_subscribers: Dict[AgentProgEvent, List[Callable[[AgentProgEvent, AgentProgContext], None]]] = {}
        self.get_fix_response = get_fix_response or self.get_response
    
    def print_to_dashboard(self, agent_prog_context: AgentProgContext, action: str=None):
        if self.config.show_dashboard:
            show_dashboard(agent_prog_context=agent_prog_context, action=action, folded=self.config.fold_dashboard)

    def subscribe_event(self, event: AgentProgEvent, callback: Callable[[AgentProgEvent, AgentProgContext], None]):
        if event not in self.event_subscribers:
            self.event_subscribers[event] = []
        self.event_subscribers[event].append(callback)
        
    def _match_cached_code(self, agent_prog_context: AgentProgContext, similar_workflows: list[WorkflowContext]):
        # hit cache if the code is same twice
        if len(similar_workflows) >= 2:
            last_first_script = similar_workflows[0].script.strip()
            last_second_script = similar_workflows[1].script.strip()
            if last_first_script == last_second_script:
                cached_node = similar_workflows[0]
                return cached_node
        return None

    def _prepare_agent_prog_context(self, workflow_context: WorkflowContext, llm_query_mode: LLMQueryMode) -> AgentProgContext:
        # 准备 Context
        script_lines = workflow_context.root.script.splitlines().copy()
        script_lines[workflow_context.workflow_context_lineno - 1] = script_lines[workflow_context.workflow_context_lineno - 1] + "  # <-- current step"

        task_description=workflow_context.root.description
        workflow_context_str="\n".join(script_lines)
        data_and_variables=print_variables(workflow_context.global_vars, workflow_context.local_vars)
        python_context=workflow_context.show_python_context(llm_query_mode=llm_query_mode)
        current_line=workflow_context.description
        return AgentProgContext(
            workflow_context=workflow_context,
            task_description=task_description,
            workflow_context_str=workflow_context_str,
            data_and_variables=data_and_variables,
            python_context=python_context,
            current_line=current_line,
            llm_query_mode=llm_query_mode,
            belief_state=self.belief_state
        )
    
    def _before_code_generation(self, agent_prog_context: AgentProgContext, executor: Callable[[str], str], workflow_prompt_set: WorkflowPromptSetOptional=None) -> CachedResult | PromptResult:
        get_core_prompt = (workflow_prompt_set and workflow_prompt_set.get_core_prompt) or self.workflow_prompt_set.get_core_prompt
        get_framework_prompt = (workflow_prompt_set and workflow_prompt_set.get_framework_prompt) or self.workflow_prompt_set.get_framework_prompt
        get_additional_info = (workflow_prompt_set and workflow_prompt_set.get_additional_info) or self.workflow_prompt_set.get_additional_info
        get_example_prompt = (workflow_prompt_set and workflow_prompt_set.get_example_prompt) or self.workflow_prompt_set.get_example_prompt
        
        workflow_context = agent_prog_context.workflow_context
        
        similar_workflows = grab_similar_nodes(workflow_context)
        
        if self.cache_mode:
            if not workflow_context.exec_res_history or workflow_context.exec_res_history[-1].is_exec_success:
                # if the process faild, don't use cache.
                reused_cached_workflow = self._match_cached_code(workflow_context, similar_workflows)
                if reused_cached_workflow is not None:
                    workflow_context.reused_cached_workflow = reused_cached_workflow # 记录
                    reuse_cached_script = "\n".join([line for line in reused_cached_workflow.script.splitlines() if not line.strip().startswith(f"{WORKFLOW_STEP_COMMENT}") and not line.strip().startswith(EXEC_RESULT_COMMENT)]) # remove many comments

                    # self.print_to_dashboard(agent_prog_context, script)

                    if need_breakpoint: breakpoint()
                    return CachedResult(cached_answer=reuse_cached_script)

        # update additional info
        additional_info = get_additional_info(agent_prog_context, similar_workflows)
        # self.print_to_dashboard(agent_prog_context, script)

        prompt_str = get_core_prompt(
            agent_prog_context=agent_prog_context,
            example_prompt=get_example_prompt() if get_framework_prompt else "",
            additional_info=additional_info,
            framework_prompt=get_framework_prompt() if get_framework_prompt else ""
        )
        if need_breakpoint: breakpoint()
        return PromptResult(prompt_str=prompt_str)

    def broadcast(self, event, agent_prog_context: AgentProgContext):
        for callback in self.event_subscribers.get(event, []):
            try:
                callback(event, agent_prog_context)
            except Exception as e:
                traceback.print_exc()
                logger.warn(e)
                continue

    def update_belief_state(self, parsed_response: AgentProgParsedResponse):
        self.belief_state.belief_state_str = parsed_response.belief_state
        self.belief_state.plan = parsed_response.plan

    def code_generation(self, workflow_context: WorkflowContext, executor: Callable[[str], str]) -> str:
        '''
        模型生成一段代码并返回。
        '''
        llm_query_mode=LLMQueryMode.CodeGeneration

        agent_prog_context: AgentProgContext = self._prepare_agent_prog_context(workflow_context, llm_query_mode)
        # update event
        self.broadcast(event=AgentProgEvent.before_code_generation, agent_prog_context=agent_prog_context)

        before_result: CachedResult | PromptResult = self._before_code_generation(agent_prog_context=agent_prog_context, executor=executor)

        if isinstance(before_result, CachedResult):
            return before_result.cached_answer
        
        elif isinstance(before_result, PromptResult):
            messages = Messages(Message(role='user', content=Prompt(before_result.prompt_str, self.workflow_prompt_set.get_images(agent_prog_context))))

            get_parsed_response = init_get_parsed_response(self.get_response, response_parser=lambda r: self.workflow_prompt_set.response_parser(AgentProgRawResponse(content=r, mode=llm_query_mode)), try_times=3, get_fix_response=self.get_fix_response)
            
            parsed_response, res = self.query(messages, get_response=get_parsed_response)
            parsed_response: AgentProgParsedResponse
            logger.info("code generation response: ")
            logger.info(res)
            self.update_belief_state(parsed_response)
            script = parsed_response.script
            assert script

            self.print_to_dashboard(agent_prog_context, script)

            if need_breakpoint: breakpoint()
            
            return script

    def _match_cached_wpc(self, workflow_context: WorkflowContext):
        if workflow_context.reused_cached_workflow is not None:
            recent_exec_res_history = workflow_context.exec_res_history[-1]
            if workflow_context.workflow_node_type == WorkflowNodeType.LOOP_FOR:
                if recent_exec_res_history.is_exec_success:
                    return WorkflowProgramCounterOperation.CONTINUE
                elif isinstance(recent_exec_res_history.error, StopIteration):
                    return WorkflowProgramCounterOperation.BREAK
            if workflow_context.workflow_node_type == WorkflowNodeType.SEQUENTIAL:
                if recent_exec_res_history.is_exec_success:
                    return WorkflowProgramCounterOperation.CONTINUE
        return None

    def _before_workflow_status_update(self, agent_prog_context: AgentProgContext, executor: Callable[[str], str], workflow_prompt_set: WorkflowPromptSetOptional=None) -> CachedResult | PromptResult:
        get_core_prompt = (workflow_prompt_set and workflow_prompt_set.get_core_prompt) or self.workflow_prompt_set.get_core_prompt
        get_framework_prompt = (workflow_prompt_set and workflow_prompt_set.get_framework_prompt) or self.workflow_prompt_set.get_framework_prompt
        get_additional_info = (workflow_prompt_set and workflow_prompt_set.get_additional_info) or self.workflow_prompt_set.get_additional_info
        get_example_prompt = (workflow_prompt_set and workflow_prompt_set.get_example_prompt) or self.workflow_prompt_set.get_example_prompt

        workflow_context = agent_prog_context.workflow_context
        # 匹配缓存
        if self.cache_mode:
            matched_cached_wpc = self._match_cached_wpc(workflow_context)
            if matched_cached_wpc is not None:
                # self.print_to_dashboard(agent_prog_context, script)
                
                if need_breakpoint: breakpoint()
                return CachedResult(cached_answer=matched_cached_wpc)
        
        additional_info = get_additional_info(agent_prog_context)
        # self.print_to_dashboard(agent_prog_context, script)

        prompt_str = get_core_prompt(
            agent_prog_context=agent_prog_context,
            example_prompt=get_example_prompt() if get_example_prompt else "",
            additional_info=additional_info,
            framework_prompt=get_framework_prompt() if get_framework_prompt else ""
        )

        if need_breakpoint: breakpoint()
        return PromptResult(prompt_str=prompt_str)

    def workflow_status_update(self, workflow_context: WorkflowContext, executor: Callable[[str], str]) -> WorkflowProgramCounterOperation:
        llm_query_mode=LLMQueryMode.WorkflowStatusUpdate

        agent_prog_context = self._prepare_agent_prog_context(workflow_context, llm_query_mode=llm_query_mode)

        before_result = self._before_workflow_status_update(agent_prog_context=agent_prog_context, executor=executor)
        
        if isinstance(before_result, CachedResult):
            return before_result.cached_answer
        
        elif isinstance(before_result, PromptResult):
            messages = Messages(Message(role='user', content=Prompt(before_result.prompt_str, self.workflow_prompt_set.get_images(agent_prog_context))))
            get_parsed_response = init_get_parsed_response(self.get_response, response_parser=lambda r: self.workflow_prompt_set.response_parser(AgentProgRawResponse(content=r, mode=llm_query_mode)), try_times=3, get_fix_response=self.get_fix_response)
            
            parsed_response, res = self.query(messages, get_response=get_parsed_response)
            parsed_response: AgentProgParsedResponse
            logger.info("wpc update response: ")
            logger.info(res)
            self.update_belief_state(parsed_response)
            thought, wpc_op = parsed_response.thought, parsed_response.wpc_op
            if thought:
                workflow_context.workflow_reflection = thought

            self.print_to_dashboard(agent_prog_context, str(wpc_op))

            if need_breakpoint: breakpoint()

            return wpc_op

    def query(self, messages: Messages, get_response=None) -> str:
        get_response = get_response or self.get_response
        messages_serialized = messages.serialize()
        response = get_response(
            messages_serialized
        )
        return response

# 全局实例
def convert_tree(workflow_tree: Dict):
    root_script = workflow_tree['script']
    # calc line number after remove.
    def remap_line_numbers(code_str: str, old_line_no: int) -> int:
        lines = code_str.strip().splitlines()
        # count lineno that is not snipe()
        removed_count = 0
        for test_lineno in range(1, old_line_no):
            if 'snipe' in lines[test_lineno - 1]:
                removed_count += 1
        # map old line no to new line no
        return old_line_no - removed_count

    def remove_snipe_str(script: str):
        return "\n".join((line for line in script.splitlines() if SNIPE_NAME not in line))

    def get_clean_tree(tree: Dict, script):
        clean_script = remove_snipe_str(script)
        if tree['state'] == 'ROOT':
            return {
                "script": clean_script,
                "task": tree['description'],
                "children": [get_clean_tree(child, script) for child in tree['children']]
            }
        else:
            new_lineno = remap_line_numbers(script, tree['context_lineno'])
            return {
            "lineno": new_lineno,
            "update_vars": tree['exec_script'],
            "duplicate_id": tree['duplicate_context_id'],
            "script": clean_script.splitlines()[new_lineno - 1].strip(),
            # "valid_script": tree['script'],
            "children": [get_clean_tree(child, script) for child in tree['children']]
        }
    
    cleaned_workflow_tree = get_clean_tree(workflow_tree, root_script)
    return cleaned_workflow_tree

def normalize_pass_statements(script: str) -> str:
    script_lines = [line for line in script.replace("pass", "").splitlines() if line.strip()]
    insert_list = []
    script_lines.append("pass")
    script_line_count = len(script_lines)
    calc_indent = lambda s: len(s) - len(s.lstrip())
    get_line = lambda n: script_lines[n - 1]
    for target_lineno in filter(lambda lineno: get_line(lineno).strip().startswith(("def ", "while ", "for ")), range(1, script_line_count + 1)):
        target_indent = calc_indent(get_line(target_lineno))
        # block end line no.
        target_end_lineno = next(filter(lambda lineno: calc_indent(get_line(lineno)) <= target_indent, range(target_lineno + 1, script_line_count + 1)), None)
        # insert pass
        # if target_end_lineno is not None:
        insert_list.append((target_end_lineno, target_indent))
    for target_end_lineno, target_indent in insert_list:
        script_lines[target_end_lineno - 1] = " " * (target_indent + 4) + "pass\n" + script_lines[target_end_lineno - 1]
    return "\n".join(script_lines)        

def preprocess_workflow(script: str):
    return '\n'.join((line) for line in script.splitlines() if line.strip() and not line.strip().startswith("#"))

def run_workflow(config: AgentProgConfig, get_openai_response: Callable[[List[Dict[str, str]]], str], task_description: str, workflow_script: str, workflow_prompt_set: StandardWorkflowPromptSet=None, inject_global_vars: Dict|None=None, workflow_callback=None, agent_prog_event_dict: Dict[AgentProgEvent, List[Callable[[AgentProgEvent, AgentProgContext], None]]]|None=None, workflow_event_dict: Dict[WorkflowEvent, List[Callable[[WorkflowEvent, WorkflowContext], None]]]|None=None):
    planning_model = AgentProgCore(config=config, get_response=get_openai_response, workflow_prompt_set=workflow_prompt_set)
    # register event for interpreter llm
    if agent_prog_event_dict is not None:
        for agent_prog_event, agent_prog_event_callbacks in agent_prog_event_dict.items():
            for agent_prog_event_callback in agent_prog_event_callbacks:
                planning_model.subscribe_event(agent_prog_event, agent_prog_event_callback)

    workflow_system = WorkflowSystem(planning_model=planning_model, workflow_callback=workflow_callback)
    
    workflow_script = preprocess_workflow(workflow_script.strip())
    logger.info(workflow_script)
    global_vars = {
        SNIPE_NAME: workflow_system.snipe,
        WORKFLOW_NAME: workflow_system.workflow
    }
    if inject_global_vars is not None:
        global_vars.update(inject_global_vars)
    local_vars = global_vars
    workflow_root = WorkflowRoot(root_script=workflow_script, workflow_system=workflow_system, description=task_description, global_vars=global_vars, local_vars=local_vars)
    workflow_system.register_plan(workflow_root)
    
    # call before execution
    if workflow_event_dict is not None:
        for before_workflow_execution_callback in workflow_event_dict.get(WorkflowEvent.before_workflow_execution, []):
            before_workflow_execution_callback(WorkflowEvent.before_workflow_execution, workflow_root)
    
    # run workflow
    workflow_system.end_workflow(workflow_root)
    if workflow_event_dict is not None:
        # call after execution
        for before_workflow_execution_callback in workflow_event_dict.get(WorkflowEvent.after_workflow_execution, []):
            before_workflow_execution_callback(WorkflowEvent.after_workflow_execution, workflow_root)

    return WorkflowResult(
        global_variables=workflow_root.global_vars,
        local_variables=workflow_root.local_vars
    )