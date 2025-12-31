import json
import math
import os
from pathlib import Path
import re
from typing import Dict, Optional
from android_world.agents import base_agent
from android_world.env import interface
from android_world.env import json_action
from agentprog.plan.code_exec.workflow.pipeline import agentprog_pipeline_core
from agentprog.plan.agentprog_utils import ToolSet, RequestMode
from agentprog.plan.code_exec.workflow.config.core_config import AgentProgConfig
from structlog import getLogger
from absl import flags

logger = getLogger(__name__)

def _extract_task_info(goal: str) -> Optional[Dict]:
    """从goal中提取对应的任务信息。
    
    Args:
        goal: 输入的任务目标字符串
        
    Returns:
        匹配到的任务元数据字典,如果未匹配则返回None
    """
    # 读取任务元数据
    task_path = os.path.join(os.path.dirname(__file__), "../task_metadata.json")
    with open(task_path, "r") as file:
        task_metadata = json.load(file)
    
    # 对每个任务模板进行精确匹配
    for task in task_metadata:
        template = task["task_template"]
        
        # 将模板转换为正则表达式模式
        # 1. 转义正则表达式特殊字符
        pattern = re.escape(template)
        
        # 2. 将{param}形式的参数替换为通配符
        pattern = re.sub(r'\\\{[^}]*\\\}', '(.*?)', pattern)
        
        # 3. 尝试匹配
        if re.match(pattern, goal):
            return task
            
    # 如果精确匹配失败，使用相似度匹配
    def normalize_text(text):
        # 移除标点符号和多余空格
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return ' '.join(text.split())
        
    def calculate_similarity(text1, text2):
        # 使用词集合的Jaccard相似度
        words1 = set(normalize_text(text1).split())
        words2 = set(normalize_text(text2).split())
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return intersection / union if union > 0 else 0
    
    # 寻找相似度最高的模板
    best_match = None
    highest_similarity = 0.6  # 设置一个最低相似度阈值
    
    for task in task_metadata:
        template = task["task_template"]
        # 将模板中的参数占位符替换为空格
        template = re.sub(r'\{[^}]*\}', '', template)
        similarity = calculate_similarity(goal, template)
        
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = task
    
    return best_match

def extract_task_info(goal: str) -> Optional[Dict]:
    with open(f"./task_{os.environ.get('EXP_NAME')}_{os.environ.get('PROCESS_ID')}.txt", "r") as f:
        task_name = f.read()

    task_path = os.path.join(os.path.dirname(__file__), "../task_metadata.json")
    with open(task_path, "r") as file:
        task_metadata = json.load(file)

    for task in task_metadata:
        if task["task_name"] == task_name:
            return task
    return _extract_task_info(goal)

class AgentProg(base_agent.EnvironmentInteractingAgent):

  def __init__(
      self,
      env: interface.AsyncEnv,
      name: str = '',
      verbose: bool = False,
  ):
    """

    Args:
      env: The environment.
      name: The agent name.
      verbose: True if the grounder should produce verbose updates.
    """
    super().__init__(env, name)
    self.name = name
    self._verbose = verbose
    self.serial_port = flags.FLAGS.console_port
    self.grpc_port = flags.FLAGS.grpc_port
    self.task_name = flags.FLAGS.tasks[0]
    self.websocket_host = os.environ.get("WEBSOCKET_HOST", "127.0.0.1")
    if 'WEBSOCKET_PORT' not in os.environ:
      raise RuntimeError('WEBSOCKET_PORT key not set.')
    self.websocket_port = os.environ["WEBSOCKET_PORT"]
    self.tool_set = ToolSet[os.environ['TOOL_SET']]

    if self.tool_set == ToolSet.mobile:
        self.base_path = Path(f"agentprog/scripts/agentprog") / os.environ.get("EXP_NAME")

    self.workflow_path: Path = self.base_path / self.task_name / f"{self.task_name}.ap"

    self.image_dir = self.base_path / self.task_name / "images"
    self.meta_info_dir = self.base_path / self.task_name / "meta_info"
    self.tensorboard_log_dir = self.meta_info_dir / "tensorboard"
    self.logging_path = self.meta_info_dir / "log.txt"

  def step(self, goal: str) -> base_agent.AgentInteractionResult:
    """See base class."""
    # grep task description from test set
    use_belief_state = True
    use_aw_locator = False
    use_ui_tars_locator = True

    ctx_logger = logger
    ctx_logger.bind(name=self.name)
    if self.name == 'agentprog_w_o_belief_state':
        ctx_logger.warn("no belief state", name=self.name)
        use_belief_state = False
    if self.name == 'agentprog_w_aw_env': # 使用 aw 的 locator
        ctx_logger.warn("use aw locator to locate", name=self.name)
        use_ui_tars_locator = False
        use_aw_locator = True

    workflow_result = agentprog_pipeline_core(AgentProgConfig(
        task_description=goal,
        workflow_path=str(self.workflow_path),
        tool_set=self.tool_set.name,
        request_mode=RequestMode.api.name,
        image_dir=self.image_dir,
        meta_info_dir=self.meta_info_dir,
        serial=f"emulator-{self.serial_port}" or "emulator-5554",
        serial_port=f"{self.serial_port}" or "5554",
        cache_mode=False,
        use_belief_state=use_belief_state,
        use_aw_locator=use_aw_locator,
        tensorboard_log_dir=self.tensorboard_log_dir,
        logging_path=self.logging_path
    ))
    answer = workflow_result.global_variables.get("answer", None)
    if answer is not None:
      action_details = {'action_type': 'answer', 'text': answer}
      self.env.execute_action(json_action.JSONAction(**action_details))

    step_data = {}
    done = True
    return base_agent.AgentInteractionResult(
        done,
        step_data,
    )
