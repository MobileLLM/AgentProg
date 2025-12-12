from __future__ import annotations
import bdb
import contextlib
from copy import deepcopy
from functools import partial
import inspect
from itertools import groupby
import itertools
import traceback
from types import FrameType
from typing import Callable, Iterable, List, Optional, Iterator, Any, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import uuid
import pandas as pd
from agentprog.all_utils import log_utils
from agentprog.all_utils.ast_utils import replace_break, replace_returns
from agentprog.all_utils.debug import need_runtime_point

logger = log_utils.get_logger(__name__)

INDENT_WIDTH = 4
SNIPE_STATE = '<snipe>' # snipe description
WORKFLOW_NAME = 'workflow'
SNIPE_NAME = 'snipe'
RET_VAL_NAME = 'ret_val'
BREAK_FLAG_NAME = 'break_flag'
CHECK_BREAK_FLAG_NAME = 'check_break_flag'
RETURN_COMMAND = f'return {RET_VAL_NAME}'
BREAK_COMMAND = f"if {BREAK_FLAG_NAME}:"f"{BREAK_FLAG_NAME} = False;""break;"
ERROR_INFO_NAME = "error_info"
IS_EXEC_SUCCESS_NAME = "is_exec_success"
EVALUATED_VALUE = "evaluating_value"
WORKFLOW_STEP_COMMENT = '# Workflow step:'
EXEC_RESULT_COMMENT = '# Execute Result:'
HIDDEN_VARS_PREFIX = '_hidden_agentprog'
MAX_RETRY_TIME = 100
MAX_LOOP_TIME = 100

class Chain:
    def __init__(self, func=lambda x: x):
        '''
        == monad return.
        '''
        self.func = func

    def __call__(self, *args, **kwds):
        return self.func(*args, **kwds)

    def __or__(self, add_func):
        '''
        ~~ monad bind, if self.func & add_func is callable.
        '''
        if callable(add_func):
            if callable(self.func):
                # 函数复合
                return Chain(lambda x: add_func(self.func(x)))
            else:
                # 函数求值
                return Chain(add_func(self.func))
        else:
            # 函数求值
            return Chain(self.func(add_func))
    
    def __radd__(self, arg):
        '''
        arg + Chain = Chain(arg)
        '''
        return self.func(arg)

class BiPart:
    def __init__(self, func):
        self.func = func
    
    def __lshift__(self, arg):
        '''
        Bipart << arg.
        '''
        return self.func(arg)

    def __rrshift__(self, arg):
        '''
        arg >> Bipart.
        '''
        return BiPart(lambda x: self.func(x, arg))

    def __rlshift__(self, arg):
        '''
        arg << Bipart.
        '''
        return BiPart(lambda x: self.func(arg, x))

    def __rshift__(self, arg):
        '''
        Bipart >> arg.
        '''
        return self.func(arg)

    def __call__(self, *args, **kwds):
        return self.func

class IterableFunctor:
    '''
    List Functor.
    '''
    def __init__(self, args: Iterable):
        self.args = args

    def fmap(self, func: Callable):
        '''
        Functor. 打开盒子，逐个应用函数，然后再装回盒子。
        '''
        return IterableFunctor(map(func, self))

    def apply(self, funcs: Iterable):
        '''
        Applicative.
        funcs 也是一个盒子，分别打开 funcs 和 self.args 盒子，调用函数，然后再装回盒子。
        '''
        return IterableFunctor(map(lambda f, a: f(a), funcs, self))
    
    def __mul__(self, func: Callable):
        return self.fmap(func=func)

    def __matmul__(self, funcs: Callable):
        return self.apply(funcs=funcs)

    def __iter__(self):
        return iter(self.args)

import ast

def is_expression(code: str) -> bool:
    try:
        node = ast.parse(code, mode='eval')
        return True  # 能用 'eval' 模式 parse，说明是表达式
    except SyntaxError:
        return False

def is_assignment(code: str) -> bool:
    try:
        node = ast.parse(code, mode='exec')
        return isinstance(node.body[0], (ast.Assign, ast.AnnAssign, ast.AugAssign))
    except Exception:
        return False

@dataclass
class WorkflowResult:
    global_variables: Dict = field(default_factory=dict)
    local_variables: Dict = field(default_factory=dict)

@dataclass
class ScriptResult:
    script: str
    error_messages: List[str]
    log_output_messages: List[str]
    full_trace_back: str
    is_exec_success: bool
    error: Exception = None
    answer: str | None = None

    @classmethod
    def deserialize(cls, data_dict: Dict):
        return ScriptResult(**data_dict)

    def format_result_str(self):
        result_list = []

        error_output = "{error_messages}\n[Full Traceback]\n{full_traceback}\n".format(
            error_messages="\n".join(self.error_messages),
            full_traceback=self.full_trace_back
        ) if self.error_messages else "\nExecution completed successfully.\n"
        result_list.append(error_output)

        log_output = "[Logs]\n{log_output_messages}\n".format(
            log_output_messages="\n".join(self.log_output_messages) if self.log_output_messages else "No log is printed."
        )
        result_list.append(log_output)

        return "\n".join(result_list)
    
    def __str__(self):
        return self.format_result_str()

class PlanExecutionError(Exception):
    def __init__(self, info: str):
        super().__init__(info)
        self.info = info

    def __str__(self):
        return f"PlanExecutionError: {self.info}"

def _find_frame(frame: FrameType, filename_prefix="<plan"):
    # 获取当前帧的局部变量
    # 从当前帧开始
    # 沿着调用栈向上查找
    while frame is not None:
        if frame.f_code.co_filename.startswith(filename_prefix):
            break  # 找到目标帧，退出循环
        frame = frame.f_back  # 继续向上移动到调用者帧
    if frame is not None:
        return frame
    else:
        # 如果遍历完整条调用栈仍未找到，抛出异常
        raise ValueError(f"No frame with filename '{filename_prefix}...' found")

def _execute_in_context(script: str, filename: str, global_vars: Dict[str, Any], local_vars: Dict[str, Any]):
    """在指定的作用域中执行代码"""
    # 配置调试工具
    def log_message(func, log_messages: List):
        '''
        用于打印日志。
        '''
        def wrap_func(*args, **kwargs):
            log_messages.append(' '.join([str(arg) for arg in args]))  # 追加日志信息
            return func(*args, **kwargs)
        return wrap_func

    local_vars['log_output'] = []
    local_vars['print'] = log_message(print, local_vars['log_output'])
    error = None
    error_messages = []
    full_trace_back = None
    is_exec_success = None

    # 为了解决列表推导式在 exec() 里面无法访问局部变量的问题，我们只能在执行前把局部变量和全局变量合并一下了。。。无奈之举，exec() 里面 dirty 的东西太多了。。。
    need_merge_scope = global_vars is not local_vars # 假设它们不是同一个的时候执行 merge 策略
    if need_merge_scope:
        merged_scope = global_vars.copy()
        globals_origin_keys = global_vars.keys()
        locals_origin_keys = local_vars.keys()
        merged_scope.update(local_vars)
    try:
        # 执行代码
        code = compile(script, filename, "exec")
        if need_merge_scope:
            exec(code, merged_scope, merged_scope)
        else:
            exec(code, global_vars, local_vars)
        is_exec_success = True
    except (bdb.BdbQuit, KeyboardInterrupt):
        exit(-1)
    except Exception as e:
        error = e
        logger.error(f"Error executing code: {traceback.format_exc()}, {type(e)}: {e}")
        if isinstance(e, RuntimeError):
            if need_runtime_point: breakpoint()
        try:
            if isinstance(e, SyntaxError) and e.filename == filename:
                error_line = script.splitlines()[e.lineno - 1] if e.lineno else "<unknown line>"
                # error_message = f"[SyntaxError] at line {e.lineno}: {e.msg}\n-> {error_line}"
                error_message = f"[SyntaxError]: {e.msg}\n-> {error_line}"
                error_messages.append(error_message)
                is_exec_success = False
                full_trace_back = traceback.format_exc()
            else:
                tb = traceback.extract_tb(e.__traceback__)  # 提取完整的错误堆栈
                for frame in tb:
                    if frame.filename == filename:  # 确保错误来自 `script`
                        error_line = script.splitlines()[frame.lineno - 1] if frame.lineno else "<unknown line>"
                        error_message = f"[{type(e).__name__}]: {e}\n-> {error_line}"
                        # error_message = f"[{type(e).__name__}] at line {frame.lineno}: {e}\n-> {error_line}"
                        error_messages.append(error_message)
                is_exec_success = False
                full_trace_back = traceback.format_exc()

                # error_messages.append(traceback.format_exc())  # 添加完整错误堆栈
        except:
            traceback.print_exc()
            if need_runtime_point: breakpoint()
    finally:
        if need_merge_scope:
            local_vars.update({ # local 是覆盖了 global 的，因此 local_vars 需要优先接管合并前同名的键，以及新产生的键
                k: v for k, v in merged_scope.items()
                if (k not in globals_origin_keys or k in locals_origin_keys) and k not in ["__builtins__"]
            })
            global_vars.update({ # global 这一边只能挑 local 剩下的变量领走
                k: v for k, v in merged_scope.items()
                if (k in globals_origin_keys and k not in locals_origin_keys)
            })
    assert is_exec_success is not None
    # 返回执行结果和日志
    log_output_messages = local_vars["log_output"]

    script_result = ScriptResult(
        script=script,
        error_messages=deepcopy(error_messages),
        log_output_messages=deepcopy(log_output_messages),
        full_trace_back=full_trace_back,
        error=error,
        is_exec_success=is_exec_success
    )

    return script_result

def get_plan_file_name(plan_id):
    '''
    必须以 <plan_ 开头。参见 _find_frame 代码。
    '''
    return f"<plan_{plan_id}>"

def filter_variables(variables: Dict):
    '''
    过滤变量键，仅保留基本类型的变量。复杂的结构（元组，列表，字典）暂时不在此范围内。也许我们得加上列表，元组，字典。
    基本类型：int, float, str, bool, None
    复杂类型：List, Dict, Tuple
    '''
    filtered_vars = {
        key: deepcopy(value) if isinstance(value, (List, Dict, Tuple)) else value
        for key, value in variables.items() 
        if (key not in ['__builtins__', WORKFLOW_NAME, 'print', 'log_output', RET_VAL_NAME, BREAK_FLAG_NAME] 
            and not callable(value)
            and (isinstance(value, (int, float, str, bool, List, Dict, Tuple)) or value is None))
            and not key.startswith(HIDDEN_VARS_PREFIX)
    }
    return filtered_vars.copy()

def compare_dicts(before, after):
    '''
    目前对于有深度的东西还无法查找。只能感知基本数据类型的变化。
    '''
    # 查找新增的键
    added = {key: after[key] for key in after.keys() - before.keys()}

    # 查找删除的键
    removed = {key: before[key] for key in before.keys() - after.keys()}

    # 查找修改的键和值
    modified = {key: after[key] for key in after if key in before and before[key] != after[key]}

    return added, removed, modified

def collect_variables_list(local_vars: dict, max_cols=10, max_rows=10, max_items=10, ignore_class_attr=(), hidden_class_attr=()):
    variables = []

    for name, value in local_vars.items():
        if name in ['__builtins__', WORKFLOW_NAME, SNIPE_NAME, 'print', 'log_output', "__warningregistry__", RET_VAL_NAME, BREAK_FLAG_NAME, CHECK_BREAK_FLAG_NAME, 'current_screenshot', 'is_exec_success', "error_info"] or name.startswith(HIDDEN_VARS_PREFIX): # 多了一个 current_screenshot
            continue
        if isinstance(value, hidden_class_attr):
            continue

        # 基本类型
        if isinstance(value, (int, float, str, bool, type(None))):
            # summary_lines.append(f"{name}: {repr(value)}")
            variables.append({
                "NL_variable": name,
                "code_variable": name,
                "type": type(value).__name__,
                "value": repr(value)
            })

        # Pandas DataFrame
        elif isinstance(value, pd.DataFrame):
            truncated_df = value.iloc[:max_rows, :max_cols]
            preview = truncated_df.to_string(index=False)
            
            # 添加省略号和总数提示
            ellipsis_info = ""
            if len(value) > max_rows or len(value.columns) > max_cols:
                ellipsis_info = f"\n... (showing {min(max_rows, len(value))} of {len(value)} rows, {min(max_cols, len(value.columns))} of {len(value.columns)} columns)"
            
            # summary_lines.append(f"{name} (DataFrame):\n{preview}{ellipsis_info}")
            variables.append({
                "NL_variable": name,
                "code_variable": name,
                "type": "DataFrame",
                "value": f"{preview}{ellipsis_info}"
            })


        # Pandas Series
        elif isinstance(value, pd.Series):
            preview = value.head(max_rows).to_string()
            ellipsis_info = ""
            if len(value) > max_rows:
                ellipsis_info = f"\n... (showing {max_rows} of {len(value)} items)"
            # summary_lines.append(f"{name} (Series):\n{preview}{ellipsis_info}")
            variables.append({
                "NL_variable": name,
                "code_variable": name,
                "type": "Series",
                "value": f"{preview}{ellipsis_info}"
            })

        # List / Tuple
        elif isinstance(value, (list, tuple)):
            preview_items = ", ".join(repr(x) for x in value[:max_items])
            ellipsis = f", ... (showing {max_items} of {len(value)} items)" if len(value) > max_items else ""
            # summary_lines.append(f"{name} ({type(value).__name__}, len={len(value)}): [{preview_items}{ellipsis}]")
            variables.append({
                "NL_variable": name,
                "code_variable": name,
                "type": type(value).__name__,
                "value": f"[{preview_items}{ellipsis}]"
            })

        # Set
        elif isinstance(value, set):
            preview_items = ", ".join(repr(x) for x in list(value)[:max_items])
            ellipsis = f", ... (showing {max_items} of {len(value)} items)" if len(value) > max_items else ""
            # summary_lines.append(f"{name} (set, len={len(value)}): {{{preview_items}{ellipsis}}}")
            variables.append({
                "NL_variable": name,
                "code_variable": name,
                "type": "set",
                "value": f"{{{preview_items}{ellipsis}}}"
            })
            
        # Dict
        elif isinstance(value, dict):
            items = list(value.items())[:max_items]
            preview_items = ", ".join(f"{repr(k)}: {repr(v)}" for k, v in items)
            ellipsis = f", ... (showing {max_items} of {len(value)} items)" if len(value) > max_items else ""
            # summary_lines.append(f"{name} (dict, len={len(value)}): {{{preview_items}{ellipsis}}}")
            variables.append({
                "NL_variable": name,
                "code_variable": name,
                "type": "dict",
                "value": f"{{{preview_items}{ellipsis}}}"
            })

        # 其他复杂对象
        else:
            continue
            if isinstance(value, tuple(ignore_class_attr)):
                # summary_lines.append(f"{name}: <{type(value).__name__}>")
                variables.append({
                    "NL_variable": name,
                    "code_variable": name,
                    "type": type(value).__name__,
                    "value": f""
                })
            else:
                # 尝试获取对象的属性预览
                try:
                    # 获取非私有属性
                    attrs = [attr for attr in dir(value) if not attr.startswith('_')]
                    if attrs:
                        # 尝试获取前几个属性的值
                        attr_previews = []
                        for attr in attrs[:5]:  # 最多显示5个属性
                            try:
                                attr_value = getattr(value, attr)
                                # 避免调用方法
                                if not callable(attr_value):
                                    attr_str = f"{attr}={repr(attr_value)}"
                                    attr_previews.append(attr_str)
                            except:
                                continue
                        
                        if attr_previews:
                            preview_text = ", ".join(attr_previews)
                            # 限制总长度为100字符
                            if len(preview_text) > 100:
                                preview_text = preview_text[:97] + "..."
                            # summary_lines.append(f"{name}: <{type(value).__name__}({preview_text})>")
                            variables.append({
                                "NL_variable": name,
                                "code_variable": name,
                                "type": type(value).__name__,
                                "value": f"{preview_text}"
                            })
                        else:
                            # summary_lines.append(f"{name}: <{type(value).__name__}>")
                            variables.append({
                                "NL_variable": name,
                                "code_variable": name,
                                "type": type(value).__name__,
                                "value": f""
                            })
                    else:
                        # summary_lines.append(f"{name}: <{type(value).__name__}>")
                        variables.append({
                            "NL_variable": name,
                            "code_variable": name,
                            "type": type(value).__name__,
                            "value": f""
                        })
                except:
                    # 如果获取属性失败，回退到简单显示
                    # summary_lines.append(f"{name}: <{type(value).__name__}>")
                    variables.append({
                        "NL_variable": name,
                        "code_variable": name,
                        "type": type(value).__name__,
                        "value": f""
                    })

    # assert isinstance(variables, List[Dict])
    return variables


def summarize_variables(local_vars: dict, max_cols=10, max_rows=10, max_items=10, ignore_class_attr=(), hidden_class_attr=()):
    summary_lines = []

    for name, value in local_vars.items():
        if name in ['__builtins__', WORKFLOW_NAME, SNIPE_NAME, 'print', 'log_output', "__warningregistry__", RET_VAL_NAME, BREAK_FLAG_NAME, CHECK_BREAK_FLAG_NAME] or name.startswith(HIDDEN_VARS_PREFIX):
            continue
        if isinstance(value, hidden_class_attr):
            continue

        # 基本类型
        if isinstance(value, (int, float, str, bool, type(None))):
            summary_lines.append(f"{name}: {repr(value)}")

        # Pandas DataFrame
        elif isinstance(value, pd.DataFrame):
            truncated_df = value.iloc[:max_rows, :max_cols]
            preview = truncated_df.to_string(index=False)
            
            # 添加省略号和总数提示
            ellipsis_info = ""
            if len(value) > max_rows or len(value.columns) > max_cols:
                ellipsis_info = f"\n... (showing {min(max_rows, len(value))} of {len(value)} rows, {min(max_cols, len(value.columns))} of {len(value.columns)} columns)"
            
            summary_lines.append(f"{name} (DataFrame):\n{preview}{ellipsis_info}")

        # Pandas Series
        elif isinstance(value, pd.Series):
            preview = value.head(max_rows).to_string()
            ellipsis_info = ""
            if len(value) > max_rows:
                ellipsis_info = f"\n... (showing {max_rows} of {len(value)} items)"
            summary_lines.append(f"{name} (Series):\n{preview}{ellipsis_info}")

        # List / Tuple
        elif isinstance(value, (list, tuple)):
            preview_items = ", ".join(repr(x) for x in value[:max_items])
            ellipsis = f", ... (showing {max_items} of {len(value)} items)" if len(value) > max_items else ""
            summary_lines.append(f"{name} ({type(value).__name__}, len={len(value)}): [{preview_items}{ellipsis}]")

        # Set
        elif isinstance(value, set):
            preview_items = ", ".join(repr(x) for x in list(value)[:max_items])
            ellipsis = f", ... (showing {max_items} of {len(value)} items)" if len(value) > max_items else ""
            summary_lines.append(f"{name} (set, len={len(value)}): {{{preview_items}{ellipsis}}}")

        # Dict
        elif isinstance(value, dict):
            items = list(value.items())[:max_items]
            preview_items = ", ".join(f"{repr(k)}: {repr(v)}" for k, v in items)
            ellipsis = f", ... (showing {max_items} of {len(value)} items)" if len(value) > max_items else ""
            summary_lines.append(f"{name} (dict, len={len(value)}): {{{preview_items}{ellipsis}}}")
        elif callable(value):
            summary_lines.append(f"{name}: <function>")
        # 其他复杂对象
        else:
            if isinstance(value, tuple(ignore_class_attr)):
                summary_lines.append(f"{name}: <{type(value).__name__}>")
            else:
                # 尝试获取对象的属性预览
                try:
                    # 获取非私有属性
                    attrs = [attr for attr in dir(value) if not attr.startswith('_')]
                    if attrs:
                        # 尝试获取前几个属性的值
                        attr_previews = []
                        for attr in attrs[:5]:  # 最多显示5个属性
                            try:
                                attr_value = getattr(value, attr)
                                # 避免调用方法
                                if not callable(attr_value):
                                    attr_str = f"{attr}={repr(attr_value)}"
                                    attr_previews.append(attr_str)
                            except:
                                continue
                        
                        if attr_previews:
                            preview_text = ", ".join(attr_previews)
                            # 限制总长度为100字符
                            if len(preview_text) > 100:
                                preview_text = preview_text[:97] + "..."
                            summary_lines.append(f"{name}: <{type(value).__name__}({preview_text})>")
                        else:
                            summary_lines.append(f"{name}: <{type(value).__name__}>")
                    else:
                        summary_lines.append(f"{name}: <{type(value).__name__}>")
                except:
                    # 如果获取属性失败，回退到简单显示
                    summary_lines.append(f"{name}: <{type(value).__name__}>")

    return "\n".join(summary_lines) if summary_lines else "No any variable now."

def grab_similar_nodes(workflow: WorkflowContext) -> list[WorkflowContext]:
    '''
    所在 workflow step 相同时被称为相似的 node，相似程度依据它们在树节点上的共同祖先的距离判断。返回最相似的 node，按照相似程度由大到小排列，若没有满足条件的返回空列表。被用于记忆和缓存机制。
    有一个 WorkflowContext.cache_node_list 也有类似的功能，不过它只返回了具有相同 parent 的 Node，也就是只有在循环里面有用。
    '''
    similar_nodes = []
    for test_workflow in workflow.workflow_system.plan_hashmap.values():
        if test_workflow.context_filename == workflow.context_filename and test_workflow.context_lineno == workflow.context_lineno and test_workflow is not workflow:
            distance = 0
            temp_workflow = workflow
            test_workflow_parents = list(test_workflow.parents)
            while temp_workflow.parent is not None and temp_workflow.parent not in test_workflow_parents:
                temp_workflow = temp_workflow.parent
                distance += 1
            
            if temp_workflow.parent in test_workflow_parents:
                # 要匹配 test_workflow 所在的 duplicate context_id
                test_duplicate_context_id = next(child for child in (temp_workflow.parent.children) if child in test_workflow_parents or child is test_workflow).duplicate_context_id
                similar_nodes.append((distance, -test_duplicate_context_id, test_workflow)) # duplicate_context_id 是越大越好
    similar_nodes.sort(key=lambda x: (x[0], x[1]))
    return [node for _, _, node in similar_nodes]

class ContextState(Enum):
    WORKFLOW = auto() # 工作流模式.
    JUMP_WPC = auto() # 跳转 wpc 的函数.
    REPLAY = auto() # replay mode，从缓存中取出代码，减少一次大模型调用。
    ROOT = auto() # 根节点，情况特殊
    SNIPE = auto() # 用于把静态代码转换为动态代码
    RECOVER = auto() # 根据 recover tree 恢复代码

class WorkflowNodeType(Enum):
    SEQUENTIAL = auto() 
    LOOP_FOR = auto()
    LOOP_WHILE = auto()
    CONDITION = auto()
    CONDITION_INLINE = auto() # 条件和语句写在同一行。
    FUNCTION = auto()
    RETURN = auto()
    WITH = auto()

class WorkflowProgramCounterOperation(Enum):
    HOLD = auto()
    RETRY = auto()
    RETURN = auto()
    BREAK = auto()
    CONTINUE = auto() 

class LLMQueryMode(Enum):
    CodeGeneration = auto()
    WorkflowStatusUpdate = auto()
    Preparation = auto()

@dataclass
class WorkflowContext:
    '''
    存储代码节点信息的结构，提供一些格式化输出的 API。
    '''
    workflow_system: 'WorkflowSystem'
    description: str
    # current_frame: FrameType # 全局唯一。
    global_vars: Dict = field(default_factory=dict)
    local_vars: Dict = field(default_factory=dict)
    # filtered_global_vars: Dict = field(default_factory=dict) # 纯净的 global & local vars。可以任意自由地复制粘贴。
    # filtered_local_vars: Dict = field(default_factory=dict) # 纯净的 local vars。可以任意自由地复制粘贴。
    context_filename: str = None # 处于哪个文件？一般而言是 parent 的文件。
    context_lineno: int = None # 处于文件的哪一个行号？这个是真实执行的文件。即 compile_workflow 编译后的文件，行号与原始 workflow 可能不同。要通过 root.line_mapping 转换。
    '''
    如果你需要匹配 workflow 的行号，请寻找 workflow_context_lineno，这大概率不是你想要的那个。
    '''
    duplicate_context_id: int = 0 # 同一个文件的同一个行号可能有多个节点，这些节点是相互冲突的，代表模型访问了多次，生成了多个版本的代码。
    parent: Optional[WorkflowContext] = None
    children: List[WorkflowContext] = field(default_factory=list)
    state: ContextState = ContextState.JUMP_WPC
    comment_id: str = None # plan_id，用于生成新的plan文件。
    think: str = None
    _script: str = ''
    exec_script: str = ''
    workflow_node_type: WorkflowNodeType = WorkflowNodeType.SEQUENTIAL
    workflow_reflection: str = ''
    reused_cached_workflow: WorkflowContext = None # 如果复用了 workflow 缓存，就放在这里，否则留空
    exec_res_history: List[ScriptResult] = field(default_factory=list)

    def __post_init__(self):
        if self.comment_id is None:
            self.comment_id = str(uuid.uuid4())
            self.script = self._script
        if self.parent is not None:
            self.parent.children.append(self)
        
        # 生成比较前后有所变化的 difference.
        # if self.parent is not None:
        #     global_added, global_removed, global_modified = compare_dicts(self.parent.filtered_global_vars, self.filtered_global_vars)
            
        #     if self.global_vars is self.local_vars:
        #         # 如果全局变量和局部变量重合了，我们就忽略局部变量。
        #         local_added, local_removed, local_modified = {}, {}, {}
        #     else:    
        #         local_added, local_removed, local_modified = compare_dicts(self.parent.filtered_local_vars, self.filtered_local_vars)
            
        #     show_if = lambda s, t='': lambda v: s.format(str(v)) if v else t
        #     comment_status = "# state: " + show_if('{}', "Nothing added, removed or modified.")("".join(
        #         map(
        #             (lambda s, g, l: 
        #                 show_if(f'{s} ''global variables: {}. ')(g) + 
        #                 show_if(f'{s} ''local variables: {}. ')(l)
        #             ),
        #             ("Added", "Removed", "Modified"), 
        #             (global_added, global_removed, global_modified), 
        #             (local_added, local_removed, local_modified)
        #         )
        #     ).strip()) + (lambda x: (" Console Print: " + repr("\n".join(x)) + ". ") if x else '')(self.global_vars['log_output']) + '-> you are now here!\n'
            comment_status = f"{WORKFLOW_STEP_COMMENT} {self.description}\n"
            self.exec_script, self.script = IterableFunctor((self.exec_script, self.script)).fmap(lambda s: s + comment_status if not s.startswith(comment_status) else '')

    @property
    def root(self) -> 'WorkflowRoot':
        return self.parent.root if self.parent is not None else self

    @property
    def parents(self):
        # as a generator
        current_comment = self
        while current_comment.parent is not None:
            yield current_comment.parent
            current_comment = current_comment.parent

    @property
    def script(self) -> str:
        '''
        返回未展开的 script. 即使原本有展开的内容，也应该收起来。
        '''
        return self._script
    
    @script.setter
    def script(self, value):
        '''
        如果需要更新赋值，记得清除所有的 child。
        更新赋值的时候，把 system 中的对应 child node 给删了。
        （TODO）有的 child 缓存可以复用，那么可以不用删，而是复用进来，但是 lineno 之类的上下文信息要同步。
        '''
        self._script = value
        # while self.children:
        #     child_plan = self.children.pop()
        #     self.comment_system.unregister_plan(child_plan)

    @property
    def workflow_context_lineno(self):
        '''
        编译前的 workflow context line no。
        '''
        return self.root.line_mapping[self.context_lineno]

    @property
    def plan_filename(self):
        '''
        本次 plan 将会生成的文件名。
        '''
        return get_plan_file_name(self.comment_id)

    @property
    def filtered_global_vars(self):
        return filter_variables(self.global_vars)

    @property
    def filtered_local_vars(self):
        return filter_variables(self.local_vars)

    @property
    def cache_node_list(self):
        '''
        按照 duplicate context id 的顺序缓存之前执行过的相同节点（如多次访问循环），按照出现先后次序返回。
        '''
        if self.parent is None:
            # 不合法的操作
            raise ValueError("No cache for this node or this node is not initialized")
        else:
            return sorted(filter(lambda x: x.context_lineno == self.context_lineno and x is not self, self.parent.children), key=lambda x: x.duplicate_context_id)

    def get_inspect_script_context(self):
        '''
        用 inspect 方法获取的 script context. 只能获取所在路的代码，不能获取支路代码。
        '''
        frame = inspect.currentframe()
        target_script = '# we are now here. the variable locals: {local_vars}'
        first_find = True
        while frame is not None:
            if frame.f_code.co_filename.startswith("<plan"):
                if first_find:
                    first_find = False
                current_plan = self.workflow_system.get_plan(frame.f_code.co_filename)
                current_script = current_plan.script
                lines = current_script.splitlines()
                current_indent_numbers = (lambda x: len(x) - len(x.lstrip()))(lines[frame.f_lineno - 1])
                indented_expand_script_content = "\n".join(' ' * current_indent_numbers + child_line for child_line in target_script.splitlines())
                lines[frame.f_lineno - 1] = indented_expand_script_content
                target_script = "\n".join(lines)
            frame = frame.f_back  # 继续向上移动到调用者帧
        return target_script
    
    def show_prettified_script(self):
        '''
        展开完全的代码。
        '''
        full_expanded_script = self.root.expand_script(current_comment=self)
        prettified_script = full_expanded_script.replace(f"{WORKFLOW_NAME}(\"\")", "").replace(f"{WORKFLOW_NAME}('')", "").replace(RETURN_COMMAND, "").replace(BREAK_COMMAND, "").strip()
        # remove empty line
        prettified_script = "\n".join([line for line in prettified_script.splitlines() if line.strip()])
        # ensure one return
        prettified_script = prettified_script.rstrip('\n') + '\n'
        # prettified_script = post_process_task(self.description) + prettified_script
        return prettified_script

    def show_python_context(self, llm_query_mode: LLMQueryMode):
        target_script_list = []
        current_context = self
        while current_context.state != ContextState.ROOT:
            target_script_list.insert(0, current_context.script)
            current_context = current_context.parent
        compiled_function_list = self.root.compiled_function_list
        return "# Program Start: \n" + "\n".join(compiled_function_list) + "\n" + "\n".join(target_script_list) + ("\n# Next lines to generate" if llm_query_mode == LLMQueryMode.CodeGeneration else "\n# Next workflow step here")

    def show_python_context_jump_wpc(self, llm_query_mode: LLMQueryMode):
        target_script_list = []
        current_context = self
        while current_context.state != ContextState.ROOT:
            target_script_list.insert(0, current_context.script)
            current_context = current_context.parent
        return ("# Program Start: \n" + "\n".join(target_script_list) + ("\n# Next lines to generate" if llm_query_mode == LLMQueryMode.CodeGeneration else "\n# Next workflow step here")).replace("workflow('<jump_wpc>', workflow_node_type='SEQUENTIAL')\n", "").replace(f"{WORKFLOW_STEP_COMMENT} <jump_wpc>\n", "")

    def expand_script(self, current_comment: WorkflowContext = None):
        '''
        从当前节点展开代码。能够获取支路的信息。
        '''
        if self._script is not None:
            script_to_expand = self._script # 需要展示当前所在的位置

            expand_script_list: List[Tuple[int, int, str]] = []

            chosen_children: List[WorkflowContext] = []
            for (duplicate_context_filename, duplicate_context_lineno), duplicate_context_children in groupby(self.children, key=lambda x: (x.context_filename, x.context_lineno)):
                duplicate_context_children_list = list(duplicate_context_children)
                # if len(duplicate_context_children_list) > 1:
                #     breakpoint()
                # 选取相同文件相同行中 id 最大的那一个
                # 特殊情况：对于所在的路需要用该路的 child。
                # 优先从 chosen map 里面提取
                chosen_child = self.workflow_system.chosen_children_map.get((duplicate_context_filename, duplicate_context_lineno), None)
                
                # 如果 chosen map 里面没有，优先取最新的。
                if chosen_child is None:
                    chosen_child = max(duplicate_context_children_list, key=lambda x:x.duplicate_context_id)
                
                chosen_children.append(chosen_child)

            for child_plan in chosen_children:
                if compare_lineno(child_plan, current_comment) in [-1, 0]: # child < current
                    child_script_lineno = child_plan.context_lineno
                    child_script_content = child_plan.expand_script(current_comment=current_comment)
                    if child_script_content is not None:
                        expand_script_list.append((child_script_lineno, 0, child_script_content))

            # Sort replacements by line number in descending order to avoid offset issues
            expand_script_list.sort(reverse=True) # 神来之笔!

            for expand_script_lineno, expand_script_priority, expand_script_content in expand_script_list:
                # 比较同一 parent 下面的 line no.
                lines = script_to_expand.split('\n')
                comment_indent_numbers = (lambda x: len(x) - len(x.lstrip()))(lines[expand_script_lineno - 1])
                indented_expand_script_content = "\n".join(' ' * comment_indent_numbers + child_line for child_line in expand_script_content.splitlines())
                lines[expand_script_lineno - 1] = indented_expand_script_content  # replace the current line
                script_to_expand = '\n'.join(lines)
        
            return script_to_expand
        else:
            return "comment_unimplemented('')"
    
    def get_workflow_tree(self, include_children: bool=True):
        '''
        获取当前节点开始的 comment tree.
        '''
        workflow_tree = {
            "description": self.description,
            "comment_id": self.comment_id,
            "plan_filename": self.plan_filename,
            "state": self.state.name,
            "script": self._script,
            "filtered_global_vars": str(self.filtered_global_vars),
            "filtered_local_vars": str(self.filtered_local_vars),
            "exec_script": self.exec_script,
            "context_lineno": self.context_lineno,
            "context_filename": self.context_filename,
            "duplicate_context_id": self.duplicate_context_id,
            "workflow_reflection": self.workflow_reflection,
            "workflow_node_type": self.workflow_node_type.name,
        }
        if include_children:
            workflow_tree.update({"children": [child.get_workflow_tree() for child in self.children]})
        return workflow_tree

    @classmethod
    def from_json(cls, json_data: dict):
        return cls(
            workflow_system=json_data.get("workflow_system"),
            description=json_data.get("description"),
            comment_id=json_data.get("comment_id"),
            state=json_data.get("state"),
            _script=json_data.get("script"),
            global_vars=json_data.get("filtered_global_vars"),
            local_vars=json_data.get("filtered_local_vars"),
            exec_script=json_data.get("exec_script"),
            context_lineno=json_data.get("context_lineno"),
            context_filename=json_data.get("context_filename"),
            duplicate_context_id=json_data.get("duplicate_context_id"),
            workflow_reflection=json_data.get("workflow_reflection"),
            workflow_node_type=WorkflowNodeType[json_data.get("workflow_node_type").removeprefix(f"{WorkflowNodeType.__name__}.")],
            children=[cls.from_json(child_json_data) for child_json_data in json_data.get("workflow_reflection", ())]
        )


def compare_lineno(comment1: WorkflowContext, comment2: WorkflowContext):
    # find the same parent where they first appear
    comment1_lineno = comment1.context_lineno
    for comment1_parent in comment1.parents:
        comment2_lineno = comment2.context_lineno
        for comment2_parent in comment2.parents:
            if comment1_parent.comment_id == comment2_parent.comment_id:
                if comment1_lineno < comment2_lineno:
                    return -1
                elif comment1_lineno > comment2_lineno:
                    return 1
                else:
                    return 0
            comment2_lineno = comment2_parent.context_lineno
        comment1_lineno = comment1_parent.context_lineno

def post_process_field(content: str, prefix: str):
    '''
    用注释的方式添加字段属性. 如 think, task.
    '''
    processed_content = content
    if not processed_content.startswith(prefix):
        processed_content = prefix + processed_content
    processed_content = processed_content.replace("\n", ' ')
    return processed_content + '\n'

post_process_think = partial(post_process_field, prefix="# think: ")
post_process_task = partial(post_process_field, prefix="# task: ")

def compile_workflow(script: str):
    calc_indent = lambda s: len(s) - len(s.lstrip())

    function_counter = 0
    def get_function_count():
        nonlocal function_counter
        function_counter += 1
        return function_counter

    function_list = []
    line_mapping = {} # 逆向查找表，从 target 查找 source

    source_line_counter = 0
    target_line_counter = 0
    def compile_line(line: str):
        nonlocal source_line_counter, target_line_counter, line_mapping
        if (line.strip().lower().startswith("for") or any(map(lambda x: x in line.lower(), ("repeat", "iterate", "traverse", "遍历", "每个", "重复")))) and line.strip().endswith((":", "：")):
            # update line mapping
            source_line_counter += 1
            target_line_counter += 1
            line_mapping[target_line_counter] = source_line_counter
            return calc_indent(line) * ' ' + f"while not {CHECK_BREAK_FLAG_NAME}() and {WORKFLOW_NAME}({repr(line.strip())}, workflow_node_type={repr(WorkflowNodeType.LOOP_FOR.name)}):"
        
        if (line.strip().lower().startswith(("while", "当")) or any(map(lambda x: x in line.lower(), ("while", "当")))) and line.strip().endswith((":", "：")):
            # update line mapping
            source_line_counter += 1
            target_line_counter += 1
            line_mapping[target_line_counter] = source_line_counter
            return calc_indent(line) * ' ' + f"while not {CHECK_BREAK_FLAG_NAME}() and {WORKFLOW_NAME}({repr(line.strip())}, workflow_node_type={repr(WorkflowNodeType.LOOP_WHILE.name)}):"
        
        elif (line.strip().lower().startswith("if") or "如果" in line):
            if line.strip().endswith((":", "：")):
                # update line mapping
                source_line_counter += 1
                target_line_counter += 1
                line_mapping[target_line_counter] = source_line_counter

                return calc_indent(line) * ' ' + f"if not {BREAK_FLAG_NAME} and {WORKFLOW_NAME}({repr(line.strip())}, workflow_node_type={repr(WorkflowNodeType.CONDITION.name)}):"
            else:
                # update line mapping
                source_line_counter += 1
                target_line_counter += 1
                line_mapping[target_line_counter] = source_line_counter

                return calc_indent(line) * ' ' + f"if not {BREAK_FLAG_NAME} and {WORKFLOW_NAME}({repr(line.strip())}, workflow_node_type={repr(WorkflowNodeType.CONDITION_INLINE.name)}): pass"
        
        elif any(map(lambda x: line.strip().lower().startswith(x), ("elif", "else", "otherwise"))):
            if line.strip().endswith((":", "：")):
                if "if" in line.strip().lower(): # is else-if
                    # update line mapping
                    source_line_counter += 1
                    target_line_counter += 1
                    line_mapping[target_line_counter] = source_line_counter

                    return calc_indent(line) * ' ' + f"elif not {BREAK_FLAG_NAME} and {WORKFLOW_NAME}({repr(line.strip())}, workflow_node_type={repr(WorkflowNodeType.CONDITION.name)}):"
                else: # pure else
                    # update line mapping
                    source_line_counter += 1
                    target_line_counter += 1
                    line_mapping[target_line_counter] = source_line_counter

                    return calc_indent(line) * ' ' + f"else:"

            else:
                if "if" in line.strip().lower(): # is else-if
                    # update line mapping
                    source_line_counter += 1
                    target_line_counter += 1
                    line_mapping[target_line_counter] = source_line_counter

                    return calc_indent(line) * ' ' + f"elif not {BREAK_FLAG_NAME} and {WORKFLOW_NAME}({repr(line.strip())}, workflow_node_type={repr(WorkflowNodeType.CONDITION_INLINE.name)}): pass"
                else: # pure else
                    # update line mapping
                    source_line_counter += 1
                    target_line_counter += 1
                    line_mapping[target_line_counter] = source_line_counter

                    return calc_indent(line) * ' ' + f"elif not {BREAK_FLAG_NAME}: workflow({repr(line.strip())}, workflow_node_type={repr(WorkflowNodeType.SEQUENTIAL.name)})"

        elif any(map(lambda x: line.strip().lower().startswith(x), ("define", "定义"))):
            # TODO: Add function
            workflow_lines = []

            # update line mapping
            source_line_counter += 1
            target_line_counter += 1
            line_mapping[target_line_counter] = source_line_counter
            workflow_lines.append(calc_indent(line) * ' ' + f"{WORKFLOW_NAME}({repr(line.strip())}, workflow_node_type={repr(WorkflowNodeType.FUNCTION.name)})")

            target_line_counter += 1
            line_mapping[target_line_counter] = source_line_counter    
            function_line = f"def func_{get_function_count()}(*args, **kwargs): # {line}"
            function_list.append(function_line + "...")
            workflow_lines.append(calc_indent(line) * ' ' + function_line)
        
            return "\n".join(workflow_lines)
        
        elif any(map(lambda x: line.strip().lower().startswith(x), ("task input", "任务输入"))):
            source_line_counter += 1
            target_line_counter += 1
            line_mapping[target_line_counter] = source_line_counter
            return calc_indent(line) * ' ' + f'# {line}'
        
        elif any(map(lambda x: line.strip().lower().startswith(x), ("task return", "任务返回"))):
            source_line_counter += 1
            target_line_counter += 1
            line_mapping[target_line_counter] = source_line_counter
            return calc_indent(line) * ' ' + f'if not {BREAK_FLAG_NAME}: return {WORKFLOW_NAME}({repr(line.strip())}, workflow_node_type={repr(WorkflowNodeType.RETURN.name)})'
        
        elif line.strip().endswith((":", "：")):
            source_line_counter += 1
            target_line_counter += 1
            line_mapping[target_line_counter] = source_line_counter

            return calc_indent(line) * ' ' + f"with {WORKFLOW_NAME}({repr(line.strip())}, workflow_node_type={repr(WorkflowNodeType.WITH.name)}):"

        else:
            source_line_counter += 1
            target_line_counter += 1
            line_mapping[target_line_counter] = source_line_counter
        
            return calc_indent(line) * ' ' + f"if not {BREAK_FLAG_NAME}: workflow({repr(line.strip())}, workflow_node_type={repr(WorkflowNodeType.SEQUENTIAL.name)})"

    compiled_workflow = '\n'.join(compile_line(line) for line in script.splitlines() if line.strip() and not line.strip().startswith("#"))

    return compiled_workflow, function_list, line_mapping

@dataclass
class WorkflowRoot(WorkflowContext):
    root_script: str = None
    # 上面的量必须初始化.
    description: str = 'Root Plan Here!'
    state: ContextState = ContextState.ROOT
    comment_id: str = "root"
    current_frame: None = None # 这是最底层的上下文了，再往上就是运行时了。
    compiled_function_list: List = field(default_factory=list)
    line_mapping: Dict = field(default_factory=dict) # 从 target code 到 source code 的 mapping

    def __post_init__(self):
        res = super().__post_init__()
        self.script = self.root_script
        self.exec_script, self.compiled_function_list, self.line_mapping = compile_workflow(self.root_script)
        logger.info("compiled_workflow: \n" + self.exec_script)
        # 添加返回值和 break_flag 全局变量.
        self.global_vars.update({
            BREAK_FLAG_NAME: False,
            RET_VAL_NAME: None
        })
        exec(f'''
def {CHECK_BREAK_FLAG_NAME}():
    global {BREAK_FLAG_NAME}
    if {BREAK_FLAG_NAME}:
        {BREAK_FLAG_NAME} = False
        return True
    else:
        return False
''', self.local_vars, self.global_vars)
        return res

def recover_comment_tree(tree_json: Dict):
    root = WorkflowRoot(
        workflow_system=WorkflowSystem()
    )

class PlanningModelInterface:
    def code_generation(self, plan: WorkflowContext, executor: Callable[[str], str]) -> str:
        raise NotImplementedError(f"{self.code_generation.__name__} is not implemented!")

    def workflow_status_update(self, plan: WorkflowContext, executor: Callable[[str], str]) -> str:
        raise NotImplementedError(f"{self.workflow_status_update.__name__} is not implemented!")

class WorkflowCallback:
    def __call__(self, global_vars, local_vars):
        raise NotImplementedError(f"{self.__name__} is not implemented!")
    
class WorkflowSystem:
    '''
    用于对 Workflow Context 进行操作。数据（Workflow Context）和操作（Workflow System）是分离的，
    '''
    def __init__(self, planning_model: PlanningModelInterface=None, recover_tree: Dict=None, workflow_callback=None):
        '''
        planning_model: 给定 plan 和执行器，planning_model 生成代码与执行器交互并拿到结果，直到完成当前 plan。（目前的情况，plan 只执行单步动作就结束。但是以后也许模型要多轮交互才结束）
        recover_tree: 如输入此参数，则从已有的 tree dict 中恢复代码，而不是调用 planning model。
        '''
        self.plan_hashmap: Dict[str, WorkflowContext] = {} # 以防万一，设计成 hashmap 方便访问。这个 hashmap 和 PlanContext 的树结构应该保持同步，同时创建同时销毁。
        self.chosen_children_map: Dict[Tuple[str, int], WorkflowContext] = {} # 通常而言是新的覆盖旧的，但是对于函数调用等需要恢复现场的情况，需要用旧的覆盖新的，对于这种特殊情况均计入本字典中。（或者也可以直接修改新旧程度的优先级？但是这样损失了一些关键的信息，因此暂时不这样做。还有别的办法吗？）
        self.model = planning_model
        self.recover_tree = recover_tree
        self.workflow_callback = workflow_callback
        if self.recover_tree is not None:
            self.plan_tree_hashmap: Dict[str, Dict] = {}
            # 从 dict 初始化 map
            def walk_in_plan_tree(current_tree: Dict):
                '''
                遍历 tree dict 并初始化 plan_dict_hashmap
                使用 filename, 
                '''
                current_tree_cp = current_tree.copy()
                current_tree_cp["children"] = [child_tree['comment_id'] for child_tree in current_tree['children']]
                self.plan_tree_hashmap[(current_tree['comment_id'])] = current_tree_cp
                for child_tree in current_tree['children']:
                    walk_in_plan_tree(child_tree)
            walk_in_plan_tree(self.recover_tree)

    def get_plan(self, plan_filename: str):
        return self.plan_hashmap[plan_filename]

    def register_plan(self, plan: WorkflowContext):
        self.plan_hashmap[plan.plan_filename] = plan
    
    def unregister_plan(self, plan: WorkflowContext):
        '''
        注销对应的 plan.
        '''
        self.plan_hashmap.pop(plan.plan_filename, None)

    def start_workflow(self, description: str, workflow_node_type: WorkflowNodeType, current_frame: FrameType, global_vars: Dict[str, Any], local_vars: Dict[str, Any]) -> WorkflowContext:
        '''
        获取一个新的 plan。
        '''
        
        logger.info("开始 plan:" + description) # 现在关闭缓存！
        
        # 初始化变量。
        new_global_vars = current_frame.f_globals
        new_local_vars = current_frame.f_locals
        
        if description == '<jump_wpc>':
            parent_plan = self.get_plan(current_frame.f_code.co_filename)
            plan_state = ContextState.JUMP_WPC
            plan = WorkflowContext(
                description=description,
                workflow_system=self,
                global_vars=new_global_vars,
                local_vars=new_local_vars,
                # filtered_global_vars=filtered_new_global_vars,
                # filtered_local_vars=filtered_new_local_vars,
                parent=parent_plan,
                context_filename=current_frame.f_code.co_filename,
                context_lineno=current_frame.f_lineno,
                duplicate_context_id=0,
                state=plan_state,
                workflow_node_type=WorkflowNodeType.SEQUENTIAL
            )
        else:

            # 静态代码跟踪模式。我们需要从静态代码中追踪到它的 parent script. 忽略 if，只考虑 for, while, def.
            # get script
            current_plan = self.get_plan(current_frame.f_code.co_filename) # 静态模式下就是根节点
            current_plan_script_lines = current_plan.exec_script.splitlines()
            # 上面两行代码实际上导致了无法跟踪在 comment system 以外的代码。
            current_script_lineno = current_frame.f_lineno
            current_script = current_plan_script_lines[current_script_lineno - 1] + '\n'
            # 寻找 new_script 的 parent script line no. 本质上就是往上搜索直到找到第一个缩进低一格的 for, while, def, if, 或者上一个对齐的语句。更新：workflow 版本里面，是搜索直到找到第一个 WORKFLOW_NAME。因为所有的 WORKFLOW_NAME 都被注册了。
            calc_indent = lambda s: len(s) - len(s.lstrip())
            current_indent = calc_indent(current_script)
            
            # 根节点的特殊性：如果上一层是空的，则判定 parent plan 为根节点。
            save_script_lineno = next(filter(
                Chain() | (lambda search_line_no: current_plan_script_lines[search_line_no - 1]) |
                (lambda search_line: (calc_indent(search_line) == current_indent) and SNIPE_NAME not in search_line),
                range(current_script_lineno, len(current_plan_script_lines) + 1)
            ), "not found") # 获取下一行需要执行的内容。

            def get_parent_script_lineno(test_script_lineno, current_indent):
                '''
                从 test_lineno 开始往上扫，直到找到可以挂的节点，或者所有节点都不能挂则返回 None。
                '''
                if test_script_lineno == 0:# 0 意味着已经搜索结束了。
                    return None
                
                test_line = current_plan_script_lines[test_script_lineno - 1]
                test_indent = calc_indent(test_line)

                if SNIPE_NAME not in test_line:
                    if test_indent < current_indent:
                        if f"{WORKFLOW_NAME}(" in test_line and test_line.strip().startswith(("for ", "if ", "elif ", "while ", "def ")): # 若第一个 indent 比当前小，且具有工作流标志，那么它一定被调用过了，那么它就是一个合适的挂树节点。对于多分支，if 和 elif 都可以挂，如果上面的行有 elif 比它小，那么这一行肯定执行过了，所以可以挂进来。
                            return test_script_lineno
                    elif test_indent == current_indent:
                        # 由于一些语句节点不挂树，需要跳过
                        # if not test_line.strip().startswith(("else")):
                        if f"{WORKFLOW_NAME}(" in test_line and test_line.strip().startswith(("for ", "if ", "while ", "def ")): # indent 相同的情形下，仍然看是否具有工作流标志，如果具有，那么就是一个合适的挂树节点。对于多分支，只能挂在 if 上，不能挂 elif，因为 elif 有可能并未被执行。
                            return test_script_lineno
                
                return get_parent_script_lineno(test_script_lineno - 1, min(test_indent, current_indent))
            
            parent_script_lineno = get_parent_script_lineno(current_script_lineno - 1, current_indent) # 从当前行的上一行开始查找
            if parent_script_lineno is not None:
                # 根据 parent script line no. 寻找到对应的 comment 节点。这个稍微有点难。对于多个符合条件的，要找到最新的那一个。
                def get_recent_plans(plan: WorkflowContext):
                    '''
                    获取到最新的 plan，对于同一行代码有多个分支的情况，只返回最新的分支。
                    '''
                    children_plans = (get_recent_plans(max(duplicate_context_children, key=lambda x: x.duplicate_context_id)) for _, duplicate_context_children in groupby(plan.children, key=lambda x: (x.context_filename, x.context_lineno)))
                    return itertools.chain((plan, ), itertools.chain.from_iterable(children_plans))
                parent_plan = next(filter(lambda child: child.context_lineno == parent_script_lineno, get_recent_plans(current_plan)), IndexError)
                if isinstance(parent_plan, Exception):
                    raise parent_plan
            else:
                parent_plan = current_plan

            max_duplicate_context_id = max(itertools.chain((-1,), (child.duplicate_context_id for child in parent_plan.children if child.context_lineno == save_script_lineno)))
            

            plan_state = ContextState.WORKFLOW
            plan = WorkflowContext(
                description=description,
                workflow_system=self,
                global_vars=new_global_vars,
                local_vars=new_local_vars,
                # filtered_global_vars=filtered_new_global_vars,
                # filtered_local_vars=filtered_new_local_vars,
                parent=parent_plan,
                context_filename=current_frame.f_code.co_filename,
                context_lineno=current_frame.f_lineno,
                duplicate_context_id=max_duplicate_context_id + 1,
                state=plan_state,
                workflow_node_type=workflow_node_type
            )

        # 沿着当前节点到根节点的路径更新 chosen child.
        greatparent_plan = plan
        while greatparent_plan:
            self.chosen_children_map[(greatparent_plan.context_filename, greatparent_plan.context_lineno)] = greatparent_plan
            greatparent_plan = greatparent_plan.parent

        self.register_plan(plan)
    
        return plan

    def end_workflow(self, workflow_context: WorkflowContext):
        '''
        执行 plan。
        '''
        def script_executor(script: str, think:str=None, workflow_callback: WorkflowCallback=None):
            '''
            更新 current script 到 python context（未做）
            '''
            workflow_callback = workflow_callback or self.workflow_callback
            concat = lambda s: lambda t: t + s
            if workflow_context.state == ContextState.JUMP_WPC:
                # 考虑如果没有赋值，就给它添加一个求值过程  
                workflow_context.exec_script, workflow_context.script = IterableFunctor((workflow_context.exec_script, workflow_context.script))\
                    .fmap(concat(script))\
                    .fmap(concat("\n" + f"{WORKFLOW_NAME}('<jump_wpc>', workflow_node_type={repr(WorkflowNodeType.SEQUENTIAL.name)})"))
                breakpoint()
            else:
                workflow_context.think = think
                # workflow_context.exec_script, workflow_context.script = IterableFunctor((workflow_context.exec_script, workflow_context.script))\
                    # .fmap(concat(script))
                # process return
                def process_exec_script(script):
                    processed_exec_script = script
                    processed_exec_script = replace_returns(processed_exec_script, ret_val_name=RET_VAL_NAME)
                    processed_exec_script = replace_break(processed_exec_script, break_flag_name=BREAK_FLAG_NAME) # 这里需要考虑，break 是作用于工作流定义的循环，还是作用于当前 script 内部定义的某个循环。如果是前者，需要修改 BREAK_FLAG_NAME 标志的值，以通知工作流跳出循环。如果是后者，break 保持原样即可。replace_break 会保留 script 内循环的 break，替换不在内循环中的 break。
                    return processed_exec_script
                workflow_context.exec_script, workflow_context.script = IterableFunctor((workflow_context.exec_script, workflow_context.script))\
                    .apply((lambda _: ("\n" + process_exec_script(script)), concat("\n" + script)))

                # if script.startswith("return"): # 原始的 return 替换策略
                #     eval_str = script.removeprefix("return").strip()
                #     workflow_context.exec_script, workflow_context.script = IterableFunctor((workflow_context.exec_script, workflow_context.script))\
                #         .apply((concat(f"\nglobals()['{RET_VAL_NAME}'] = {eval_str}"), concat(script)))
                # else:
                #     # default action
                #     workflow_context.exec_script = script
                #     workflow_context.script += "\n" + script

                workflow_context.script = workflow_context.script.replace("\n\n", "\n")

            exec_result = _execute_in_context(workflow_context.exec_script, filename=workflow_context.plan_filename, global_vars=workflow_context.global_vars, local_vars=workflow_context.local_vars)
            if callable(workflow_callback):
                workflow_callback(workflow_context.global_vars, local_vars=workflow_context.local_vars)
            workflow_context.global_vars.update({
                IS_EXEC_SUCCESS_NAME: exec_result.is_exec_success,
            })
            if not exec_result.is_exec_success:
                error_info = "\n".join(exec_result.error_messages)
                workflow_context.global_vars.update({
                    ERROR_INFO_NAME: error_info,
                })
                workflow_context.script += "\n" + f"{EXEC_RESULT_COMMENT} {repr(error_info)}"
                if workflow_context.state == ContextState.JUMP_WPC:
                    _execute_in_context('\n' * (len(workflow_context.exec_script.splitlines()) - 1) + f"{WORKFLOW_NAME}('<jump_wpc>', workflow_node_type={repr(WorkflowNodeType.SEQUENTIAL.name)})", filename=workflow_context.plan_filename, global_vars=workflow_context.global_vars, local_vars=workflow_context.local_vars)
            else:
                # workflow_context.script += "\n" + f"{EXEC_RESULT_COMMENT} Success."
                workflow_context.global_vars.pop(ERROR_INFO_NAME, None)
            return exec_result
                    
        if workflow_context.state == ContextState.JUMP_WPC:
            # 获取上下文并调用模型
                # clear_and_logger.info(plan.root.description, plan.show_prettified_script(), "global vars: ", plan.filtered_global_vars, "local vars:", plan.filtered_local_vars, "current script: ", script)

            if self.model is not None:
                generated_code = self.model.code_generation(workflow_context, script_executor)
                # 这个 generated code 暂时没用了，因为 model 会把代码保存在 plan 里面。
            else:
                raise AttributeError("No Planning Model Found in WorkflowSystem!")
            workflow_context.state = ContextState.REPLAY
        elif workflow_context.state == ContextState.REPLAY:
            # 这个没什么用了。本来是想命中缓存的时候避免 query 大模型，现在不用缓存，每次都必 query 大模型。
            _execute_in_context(workflow_context.exec_script, workflow_context.plan_filename, workflow_context.global_vars, workflow_context.local_vars)
        elif workflow_context.state == ContextState.ROOT:
            return _execute_in_context(workflow_context.exec_script, workflow_context.plan_filename, workflow_context.global_vars, workflow_context.local_vars)
        
        elif workflow_context.state == ContextState.WORKFLOW:
            match workflow_context.workflow_node_type:
                case WorkflowNodeType.SEQUENTIAL:
                    # retry time
                    for _ in range(MAX_RETRY_TIME):
                        generated_code = self.model.code_generation(workflow_context, script_executor)
                        exec_res = script_executor(generated_code, workflow_callback=self.workflow_callback)
                        workflow_context.exec_res_history.append(exec_res)
                        wpc_operation = self.model.workflow_status_update(workflow_context, script_executor)
                        match wpc_operation:
                            case WorkflowProgramCounterOperation.CONTINUE:
                                return True
                            # HOLD 和 RETRY 的区别在哪，行为上好像是一样的。！现在我知道他们的区别了。HOLD 表示当前任务尚未完成，但是目前的做法没有错误。而 RETRY 则表示目前的做法不能完成任务，思路上就是错的，需要更换策略。 
                            case WorkflowProgramCounterOperation.HOLD:
                                continue
                            case WorkflowProgramCounterOperation.RETRY:# 这里要改成抛出异常，更换策略。
                                if exec_res.is_exec_success:
                                    exec_res.is_exec_success = False
                                    exec_res.error_messages.append("尽管代码执行成功，代码审核认为这段代码存在问题，并未达到其目的或者功能，因此请你根据代码审核的反馈信息重新生成代码。")
                                continue
                            case WorkflowProgramCounterOperation.BREAK:
                                workflow_context.global_vars[BREAK_FLAG_NAME] = True # 设置 breakflag name。
                                return False
                    return True
                    # raise RuntimeErrLOOP_FOR"Sequential Retry times more than max time ({MAX_RETRY_TIME} times)")
                case WorkflowNodeType.LOOP_WHILE | WorkflowNodeType.LOOP_FOR:
                    # 使用迭代器
                    count = 0
                    for _ in range(MAX_LOOP_TIME):
                        generated_code = self.model.code_generation(workflow_context, script_executor)
                        exec_res = script_executor(generated_code, workflow_callback=self.workflow_callback)
                        workflow_context.exec_res_history.append(exec_res)
                        wpc_operation = self.model.workflow_status_update(workflow_context, script_executor)
                        match wpc_operation:
                            case WorkflowProgramCounterOperation.CONTINUE:
                                count += 1
                                return True
                            # HOLD 和 RETRY 的区别在哪，行为上好像是一样的
                            case WorkflowProgramCounterOperation.HOLD:
                                continue
                            case WorkflowProgramCounterOperation.RETRY:
                                continue
                            case WorkflowProgramCounterOperation.BREAK:
                                return False
                    raise RuntimeError(f"Loop times more than max time ({MAX_LOOP_TIME} times)")
                    
                case WorkflowNodeType.CONDITION | WorkflowNodeType.CONDITION_INLINE:
                    generated_code = self.model.code_generation(workflow_context, script_executor)
                    exec_res = script_executor(generated_code, workflow_callback=self.workflow_callback)
                    workflow_context.exec_res_history.append(exec_res)
                    wpc_operation = self.model.workflow_status_update(workflow_context, script_executor)
                    match wpc_operation:
                        case WorkflowProgramCounterOperation.CONTINUE:
                            return True
                        case WorkflowProgramCounterOperation.BREAK:
                            return False

                case WorkflowNodeType.WITH:
                    class WithWrapper:
                        def __init__(self):
                            pass
                        def __enter__(self):
                            return self
                        def __exit__(self, type, value, trace):
                            return False  # 如果返回 True，则异常不会向上传播
                    return WithWrapper()

                case WorkflowNodeType.FUNCTION:
                    # Nothing to do.
                    pass

                case WorkflowNodeType.RETURN:
                    for _ in range(MAX_RETRY_TIME):
                        generated_code = self.model.code_generation(workflow_context, script_executor)
                        exec_res = script_executor(generated_code, workflow_callback=self.workflow_callback)
                        workflow_context.exec_res_history.append(exec_res)
                        wpc_operation = self.model.workflow_status_update(workflow_context, script_executor)
                        match wpc_operation:
                            case WorkflowProgramCounterOperation.CONTINUE | WorkflowProgramCounterOperation.RETURN:
                                rv = workflow_context.global_vars[RET_VAL_NAME]
                                workflow_context.global_vars[RET_VAL_NAME] = None # 清空 RET_VAL_NAME
                                return rv
                            
                            # HOLD 和 RETRY 的区别在哪，行为上好像是一样的
                            case WorkflowProgramCounterOperation.HOLD | WorkflowProgramCounterOperation.RETRY:
                                continue

                    return None
            pass
        elif workflow_context.state == ContextState.SNIPE:
            pass

    @contextlib.contextmanager
    def plan(self, description: str):
        current_frame = _find_frame(inspect.currentframe())
        global_vars = current_frame.f_globals
        local_vars = current_frame.f_locals
        plan_ctx = self.start_workflow(description, current_frame, global_vars, local_vars)
        try:
            yield plan_ctx
        finally:
            self.end_workflow(plan_ctx)

    def workflow(self, description, workflow_node_type: str, workflow_callback: WorkflowCallback=None):
        workflow_callback = workflow_callback or self.workflow_callback
        logger.info("workflow_node_type: " + workflow_node_type)
        current_frame = _find_frame(inspect.currentframe())
        if callable(workflow_callback):
            workflow_callback(current_frame.f_globals, current_frame.f_locals)
        plan_ctx = self.start_workflow(description, WorkflowNodeType[workflow_node_type], current_frame, None, None) # 初始化 workflow。
        workflow_result = self.end_workflow(plan_ctx)
        return workflow_result # 对于 for 节点，是迭代是否结束；对于 if 节点，是返回布尔值。对于顺序节点而言，这个值并不重要。

    def snipe(self, snipe_callback=None):
        '''
        记录上下文。
        '''
        current_frame = _find_frame(inspect.currentframe())
        
        if callable(snipe_callback):
            snipe_callback(current_frame)
        plan_ctx = self.start_workflow(SNIPE_STATE, current_frame, None, None)
        self.end_workflow(plan_ctx)
