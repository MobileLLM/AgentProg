from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from dataclasses import dataclass
import traceback
from absl import flags
from pathlib import Path
import time
import yaml
from agentprog.all_utils.debug import need_breakpoint 
from agentprog.plan.agentprog_utils import ToolSet
import subprocess
import signal
import os
import threading
import time
from dotenv import load_dotenv; load_dotenv()

def print_with_log(*args, **kwargs):
    global log_path
    Path(log_path).parent.mkdir(exist_ok=True, parents=True)
    print(*args, **kwargs)
    with log_path.open('a') as f:
        print(*args, **kwargs, file=f)

@dataclass
class TaskResult:
    is_success: bool
    is_interrupted: bool = False # Whether manually interrupted
    error: Exception = None
    trace: str = None

def run_task(run_cmd: str, timeout: int):
    '''
    TODO: Rewrite this script to ensure compatibility with multi-processing.
    Otherwise, currently, processes can only be started one by one.
    '''
    process = None
    timer = None
    error = None
    def timeout_handler():
        """Timeout handler function"""
        nonlocal error
        if process:
            print_with_log(f"\nCommand timed out ({timeout}s), forcing termination...")
            error = f"Command timed out ({timeout}s), forced termination"
            try:
                # Try graceful termination first
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                time.sleep(2)  # Give 2 seconds for graceful exit
                
                if process.poll() is None:  # If still not exited
                    print_with_log("Force killing process...")
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass  # Process no longer exists
    
    try:
        # Start child process, output to terminal in real-time
        process = subprocess.Popen(
            run_cmd,
            shell=True,
            stdout=None,  # Output directly to terminal
            stderr=None,  # Output directly to terminal
            preexec_fn=os.setsid  # Create new process group
        )
        
        # If timeout is set, start timer
        if timeout and not need_breakpoint:
            timer = threading.Timer(timeout, timeout_handler)
            timer.start()
        
        # Wait for process to complete
        return_code = process.wait()
        
        # Cancel timer
        if timer:
            timer.cancel()
        
        print_with_log(f"\nProcess ended, return code: {return_code}")
        # return return_code == 0
        return TaskResult(
            is_success=return_code == 0,
            error=error
        )
        
    except KeyboardInterrupt:
        print_with_log("\nUser interrupted, terminating process...")
        if process:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.wait()
        if timer:
            timer.cancel()
        return TaskResult(
            is_success=False,
            is_interrupted=True
        )
        
    except Exception as e:
        print_with_log(f"Error executing command: {e}")
        if process:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        if timer:
            timer.cancel()
        return TaskResult(
            is_success=False,
            is_interrupted=True,
            error=e,
            trace=traceback.format_exc()
        )

@dataclass
class ProcessConfig:
    process_id: int
    console_port: int
    grpc_port: int
    websocket_port: int = 6666
    websocket_host: str = "127.0.0.1"

    @classmethod
    def from_dict(cls, process_id: int, data: dict):
        return ProcessConfig(**data, process_id=process_id)

def setup_env(process_config: ProcessConfig):
    os.system(f"adb -s emulator-{process_config.console_port} forward tcp:{process_config.websocket_port} tcp:6666")

def run_tasks(tool_set: str, tasks: list, process_num: int, process_config: ProcessConfig):
    process_id = process_config.process_id
    exp_name = os.environ['EXP_NAME']
    for task in tasks[process_id::process_num]:
        task: str
        task_log_path = Path("agentprog/scripts/agentprog") / exp_name / task / "full_log.txt"
        task_log_path.parent.mkdir(parents=True, exist_ok=True)
        print_with_log(f"[INFO] running: {task}")
        task_fail_path = save_path / f"{task}.failed"
        if (save_path / f"{task}_0.pkl.gz").exists():
            (save_path / f"{task}_0.pkl.gz").unlink()
        if task_fail_path.exists():
            task_fail_path.unlink()
        run_cmd = f"EXP_NAME=\"{exp_name}\" PROCESS_NUM={process_num} PROCESS_ID={process_id} WEBSOCKET_HOST={process_config.websocket_host} WEBSOCKET_PORT={process_config.websocket_port} TOOL_SET={tool_set} python run.py --tasks={task} --agent_name={agent_name} --checkpoint_dir={save_path} --console_port={process_config.console_port} --grpc_port={process_config.grpc_port}" + (f" 2>&1 | tee -a {log_path} -a {task_log_path}" if not need_breakpoint else "")
        print_with_log(run_cmd)
        result = run_task(run_cmd=run_cmd, timeout=task_timeout)
        if not result.is_success and not result.is_interrupted and not need_breakpoint:
            task_fail_path.parent.mkdir(parents=True, exist_ok=True)
            task_fail_path.write_text(f"task {task} failed! \nError: {result.error} \nTraceback:\n {result.trace}")
        time.sleep(3)

if os.environ.get("LONG_TASKS", "0") == "1":
    task_path = Path("android_world/tasks_long.txt")
else:
    task_path = Path("android_world/tasks.txt")

tasks = task_path.read_text()

task_timeout = 4*60*60
# os.environ['EXP_NAME'] = '0920_w_uitars15'
# os.environ['CONSOLE_PORT'] = '5556'
# os.environ['GRPC_PORT'] = '8556'
# os.environ['WEBSOCKET_PORT'] = '9999'
# LONG_TASKS=0
node_path = os.environ.get("NODE_PATH", "node_config.yaml")
config_data = yaml.safe_load(Path(node_path).read_text(encoding='utf-8'))
process_configs = [ProcessConfig.from_dict(process_id, process_dict) for process_id, process_dict in enumerate(config_data['processes'])]
process_num = len(process_configs)
process_id = int(os.environ["PROCESS_ID"])
tool_set = os.getenv("TOOL_SET", ToolSet.mobile.name)
agent_name = os.getenv('AGENT_NAME', 'agentprog')
assert agent_name in ['agentprog', 'agentprog_w_o_belief_state', 'agentprog_w_aw_env']
# agent_name = 'agentprog'
# agent_name = 'agentprog_w_o_belief_state'
# agent_name = 'agentprog_w_aw_env'

save_path = Path(f"runs/{agent_name}/") / os.environ["EXP_NAME"]
log_path = Path(f"{save_path}_{process_id}.log")
unfinished_tasks = [task for task in tasks.split("\n") if task and not task.startswith("# ") and not (save_path / f"{task}_0.pkl.gz").exists() and not (save_path / f"{task}.failed").exists()]
process_config = process_configs[process_id]
setup_env(process_config=process_config)
run_tasks(tool_set, unfinished_tasks, process_num, process_config=process_config)
