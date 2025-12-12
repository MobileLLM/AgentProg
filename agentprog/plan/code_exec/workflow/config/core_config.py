from __future__ import annotations
from dataclasses import dataclass
import dataclasses
from agentprog.plan.agentprog_utils import RequestMode, ToolSet

@dataclass
class AgentProgConfig:
    workflow_path: str
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

    @classmethod
    def get_field_names(cls):
        return [field.name for field in dataclasses.fields(cls)]
    
    @classmethod
    def get_field_default_value(cls):
        return {field.name: field.default for field in dataclasses.fields(cls)}

if __name__ == "__main__":
    print(AgentProgConfig.get_field_default_value())