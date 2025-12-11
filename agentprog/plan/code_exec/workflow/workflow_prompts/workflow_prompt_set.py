from typing import Callable, List, Protocol
from PIL import Image
from dataclasses import dataclass, field

from agentprog.plan.code_exec.general_prompts.framework_prompts.mobile_prompt import get_mobile_prompt
from agentprog.plan.code_exec.workflow.workflow_prompts.core_prompts.cot_core_prompt_w_belief_state import get_cot_core_prompt_w_belief_state
from agentprog.plan.agentprog_utils import AgentProgContext, AgentProgParsedResponse, AgentProgRawResponse, _parse_script_only, _parse_script_w_belief_state, _parse_thought_script, print_variables
from agentprog.plan.code_exec.workflow.workflow_prompts.core_prompts.cot_core_prompt import get_cot_core_prompt
from agentprog.plan.code_exec.workflow.workflow_prompts.example_prompts.mobile_example_prompt import get_mobile_example_cot_prompt
from agentprog.plan.code_exec.workflow.workflow_prompts.additional_info_prompts import get_additional_info_for_code_generation_mobile, get_additional_info_for_workflow_status_update_mobile
from agentprog.plan.workflow_utils import LLMQueryMode

class GetCorePrompt(Protocol):
    def __call__(self, interpreter_llm_context: AgentProgContext, example_prompt: str, additional_info: str, framework_prompt: str="") -> str:
        raise NotImplementedError("get prompt function doesn't set!")

class GetAdditionalInfoPrompt(Protocol):
    def __call__(self, interpreter_llm_context: AgentProgContext, similar_workflows=None) -> str:
        raise NotImplementedError("get prompt function doesn't set!")

class GetPreparationStepPrompt(Protocol):
    def __call__(interpreter_llm_context: AgentProgContext, example_prompt: str, additional_info: str, framework_prompt: str=""):
        raise NotImplementedError("get prompt function doesn't set!")

class UpdateBeliefStatePrompt(Protocol):
    def __call__(interpreter_llm_context: AgentProgContext, example_prompt: str, additional_info: str, framework_prompt: str=""):
        raise NotImplementedError("get prompt function doesn't set!")

class GetFrameworkPrompt(Protocol):
    def __call__(self) -> str:
        raise NotImplementedError("get prompt function doesn't set!")

class GetExamplePrompt(Protocol):
    def __call__(self) -> str:
        raise NotImplementedError("get prompt function doesn't set!")

ResponseParserType = Callable[[AgentProgRawResponse], AgentProgParsedResponse]
GetImagesType = Callable[[AgentProgContext], List[Image.Image | str]]

@dataclass
class WorkflowPromptSetBase:
    get_core_prompt: GetCorePrompt = None
    get_framework_prompt: GetFrameworkPrompt = None
    get_additional_info: GetAdditionalInfoPrompt = None
    response_parser: ResponseParserType = None
    get_example_prompt: GetExamplePrompt = None
    get_images: GetImagesType = lambda _: []

@dataclass
class WorkflowPromptSetOptional(WorkflowPromptSetBase):
    get_core_prompt: GetCorePrompt = None
    get_framework_prompt: GetFrameworkPrompt = None
    get_additional_info: GetAdditionalInfoPrompt = None
    response_parser: ResponseParserType = None
    get_example_prompt: GetExamplePrompt = None
    get_images: GetImagesType = None

@dataclass
class StandardWorkflowPromptSet(WorkflowPromptSetBase):
    update_belief_state: UpdateBeliefStatePrompt = None
    get_additional_info_for_code_generation: GetAdditionalInfoPrompt = None
    get_additional_info_for_workflow_status_update: GetAdditionalInfoPrompt = None
    
    def __post_init__(self):
        self.get_additional_info = self.route_additional_info_prompt
    
    def route_additional_info_prompt(self, interpreter_llm_context: AgentProgContext, similar_workflows=None):
        match interpreter_llm_context.llm_query_mode:
            case LLMQueryMode.CodeGeneration:
                return self.get_additional_info_for_code_generation(interpreter_llm_context, similar_workflows)
            case LLMQueryMode.WorkflowStatusUpdate:
                return self.get_additional_info_for_workflow_status_update(interpreter_llm_context, similar_workflows)

@dataclass
class MobileWorkflowPromptSet(StandardWorkflowPromptSet):
    get_framework_prompt: GetFrameworkPrompt = get_mobile_prompt
    get_additional_info_for_code_generation: GetAdditionalInfoPrompt =  get_additional_info_for_code_generation_mobile
    get_additional_info_for_workflow_status_update: GetAdditionalInfoPrompt =  get_additional_info_for_workflow_status_update_mobile

@dataclass
class CoTMobileWorkflowPromptSet(MobileWorkflowPromptSet):
    get_core_prompt: GetCorePrompt = get_cot_core_prompt
    response_parser: ResponseParserType = _parse_thought_script
    get_example_prompt: GetExamplePrompt = get_mobile_example_cot_prompt

@dataclass
class CoTMobileWorkflowPromptSetWithBeliefState(MobileWorkflowPromptSet):
    get_core_prompt: GetCorePrompt = get_cot_core_prompt_w_belief_state
    response_parser: ResponseParserType = _parse_script_w_belief_state
    get_example_prompt: GetExamplePrompt = get_mobile_example_cot_prompt
