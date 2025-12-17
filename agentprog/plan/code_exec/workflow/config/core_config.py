from __future__ import annotations
from dataclasses import dataclass, field
import dataclasses
from datetime import datetime
from agentprog.all_utils.general_utils import InitResponseArgs, init_get_response
from agentprog.plan.agentprog_utils import RequestMode, ToolSet

@dataclass
class AgentProgConfig:
    workflow_path: str = field(default_factory=lambda: f"outputs/{datetime.now().strftime('%Y%m%d%H%M%S')}.ap")
    task_description: str = ""
    request_mode: str = RequestMode.api.name
    tool_set: str = ToolSet.mobile.name
    image_dir: str = "outputs/images"
    meta_info_dir: str = "outputs/meta_info"
    serial: str = 'emulator-5554'
    serial_port: str = '5554'
    cache_mode: bool = False
    use_belief_state: bool = True
    use_aw_locator: bool = False
    tensorboard_log_dir: str = None
    logging_path: str = None
    show_dashboard: bool = True
    fold_dashboard: bool = False
    workflow_model_args: InitResponseArgs = field(default_factory=lambda: InitResponseArgs(model=None))
    executor_model_args: InitResponseArgs = field(default_factory=lambda: InitResponseArgs(model=None))

    def __post_init__(self):
        default_model_args = InitResponseArgs(
            model='vertex_ai/gemini-2.5-pro', 
            record_completion_statistics=True, 
            tensorboard_log_dir=self.tensorboard_log_dir,
            completion_kwargs={
                "temperature": 0.6,
                "stream": False,
                "thinking": {"type": "enabled"}
            }
        )
        
        self.workflow_model_args.update_args(default_model_args)
        self.executor_model_args.update_args(default_model_args)
        self.get_workflow_response = init_get_response(self.workflow_model_args)
        self.get_executor_response = init_get_response(self.executor_model_args)

    @classmethod
    def get_field_names(cls):
        return [field.name for field in dataclasses.fields(cls)]
    
    @classmethod
    def get_field_default_value(cls):
        return {field.name: field.default if not isinstance(field.default, dataclasses._MISSING_TYPE) else field.default_factory() for field in dataclasses.fields(cls)}

if __name__ == "__main__":
    print(AgentProgConfig.get_field_default_value())