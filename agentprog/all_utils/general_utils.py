from __future__ import annotations
from bdb import BdbQuit
from collections.abc import Callable
import dataclasses
from functools import partial
import os
import time
import traceback
from types import FrameType
import requests
from copy import deepcopy
from pathlib import Path
from PIL import Image
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Protocol
from uuid import uuid4
import io
import requests
import base64
from tenacity import before_sleep_log, retry, wait_fixed, stop_after_attempt
from agentprog.all_utils.log_utils import structlog, logging
from dotenv import load_dotenv; load_dotenv(override=True)
from tenacity import RetryCallState
import tenacity
import atexit
from agentprog.all_utils import log_utils

# 设置日志配置
retry_logger = log_utils.get_logger(__name__)
# retry_logger.setLevel(level=logging.INFO)

SerializeMode = Literal["openai", "qwen"] # 目前支持 openai，qwen


def log_retry_error_with_traceback(retry_state: RetryCallState, handler=print):
    """记录错误和完整traceback"""
    if retry_state.outcome.failed:
        exception = retry_state.outcome.exception()        
        # 获取完整的 traceback
        tb_str = ''.join(traceback.format_exception(
            type(exception), 
            exception, 
            exception.__traceback__
        ))
        handler(tb_str)

def image_to_base64(image: Image.Image):
    # 将 PIL.Image 转换为 base64
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")  # 保存到内存缓冲区，格式可改为 JPEG 等
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    buffer.close()  # 释放缓冲区
    return image_base64

    # 现在 image_base64 是一个 base64 字符串

def base64_to_image(image_base64):
    # 解码 base64 字符串为字节数据
    image_data = base64.b64decode(image_base64)
    # 使用 BytesIO 创建字节流，再用 PIL.Image 打开
    image = Image.open(io.BytesIO(image_data))
    return image

def make_system(*args):
    prompt = Prompt()
    for arg in args:
        prompt.append(arg)
    return Message("system", prompt)

def make_user(*args):
    prompt = Prompt()
    for arg in args:
        prompt.append(arg)
    return Message("user", prompt)

def make_assistant(*args):
    prompt = Prompt()
    for arg in args:
        prompt.append(arg)
    return Message("assistant", prompt)

IMAGE_START = "{|{|"
IMAGE_END = "|}|}"

from typing import Any

class RichStr(str):
    RAW: str = "raw"
    def __new__(cls, value: str, raw: Any):
        # 创建 str 对象并附加额外的 raw 信息
        obj = str.__new__(cls, value)
        setattr(obj, RichStr.RAW, raw)
        return obj
    
    def get_raw(self):
        return self.raw

@dataclass
class Prompt:
    prompt_template: str = ""
    images: dict[str, Image.Image | Path | str] = field(default_factory=dict)
    strict_mode: bool = True # 若非严格模式，尽可能解析所有潜在的 images。

    def append(self, content: str | Image.Image | Prompt) -> Prompt:
        match content:
            case str(content):
                if content.startswith(IMAGE_START) and content.endswith(IMAGE_END):
                    self.prompt_template += content
                    image_id = content.removeprefix(IMAGE_START).removesuffix(IMAGE_END)
                    self.images[image_id] = image_id
                else:
                    self.prompt_template += content
            case Image.Image():
                image_id = str(uuid4())
                self.prompt_template += IMAGE_START + image_id + IMAGE_END
                self.images[image_id] = content.copy()
            case Prompt(prompt_template, images):
                self.prompt_template += deepcopy(prompt_template)
                self.images.update(deepcopy(images))
            case _:
                raise TypeError(f"Unsupported Type of content: {type(content)}")
        return self

    def replace(self, key, value: str | Image.Image | Prompt) -> Prompt:
        '''
        注意！目前是浅拷贝。
        '''
        match value:
            case str():
                self.prompt_template = self.prompt_template.replace(key, value)
            case Image.Image():
                image_id = str(uuid4())
                self.prompt_template = self.prompt_template.replace(key, image_id)
                self.images[image_id] = value.copy()
            case Prompt(prompt_template, images):
                self.prompt_template = self.prompt_template.replace(key, deepcopy(prompt_template))
                self.images.update(deepcopy(images))
            case _:
                raise TypeError(f"Unsupported Type of value: {type(value)}")
        return self
        
    def serialize(self, mode: SerializeMode="openai") -> List[Dict]:
        content_list = []
        
        prompt_remaining = self.prompt_template
        # 处理图片和文字
        while IMAGE_START in prompt_remaining:
            current_content, prompt_remaining = prompt_remaining.split(IMAGE_START, maxsplit=1)
            content_list.append({"type": "text", "text": current_content})
            
            current_image_id, prompt_remaining = prompt_remaining.split(IMAGE_END, maxsplit=1)
            if current_image_id in self.images:
                future_image = self.images[current_image_id]
                if isinstance(future_image, (Image.Image)):
                    current_image = future_image
                elif isinstance(future_image, (Path, str)):
                    current_image = Image.open(future_image)
                image_base64 = image_to_base64(current_image)
                match mode:
                    case 'openai':
                        content_list.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"} })
                    case 'qwen':
                        content_list.append({"type": "image_url", "image_url": f"data:image/png;base64,{image_base64}"})
                    case _:
                        raise ValueError(f"invalid type of mode: {mode}")
            else:
                current_image = None
                if not self.strict_mode:
                    retry_logger.warning("image not found in current prompt, trying to match any possible image(in unstrict mode)", current_image_id=current_image_id)
                    if Path(current_image_id).exists():
                        try:
                            current_image = Image.open(future_image)
                        except Exception as e:
                            traceback.print_exc()
                            print(e)
                
                if current_image is None:
                    retry_logger.warning("image not found in current prompt, using image id directly", current_image_id=current_image_id)
                    content_list.append({"type": "text", "text": current_image_id})

        if prompt_remaining:
            content_list.append({"type": "text", "text": prompt_remaining})
        # for image_id, future_image in self.images.items():
        #     if isinstance(future_image, Image.Image):
        #         image = future_image
        #     elif isinstance(future_image, (Path, str)):
        #         image = Image.open(future_image)
        #     current_content, prompt_remaining = prompt_remaining.split(IMAGE_START + image_id + IMAGE_END)
        #     content_list.append({"type": "text", "text": current_content})

        #     image_base64 = image_to_base64(image)
        #     content_list.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"} })
        # if prompt_remaining:
        #     content_list.append({"type": "text", "text": prompt_remaining})

        return content_list

    def __str__(self):
        return self.prompt_template

@dataclass
class Message:
    role: Literal["system", "user", "assistant"]
    content: Prompt

    def append(self, item: str | Image.Image | Prompt) -> Message:
        self.content.append(item)
        return self

    def serialize(self, mode: SerializeMode = 'openai'):
        '''
        Convert to standard OpenAI request method.
        '''
        return {"role": self.role, "content": self.content.serialize(mode=mode)}

    def __str__(self):
        return str({"role": self.role, "content": str(self.content)})

@dataclass
class Messages:
    messages: List[Message] = field(default_factory=list)

    def __post_init__(self):
        '''
        假若 messages 误传成了 message，也可纠正过来。
        '''
        if isinstance(self.messages, Message):
            self.messages = [self.messages]

    def append(self, item: Message) -> Messages:
        self.messages.append(item)
        return self

    def serialize(self, mode: SerializeMode="openai"):
        '''
        方便 LLM Query.
        '''
        return [message.serialize(mode) for message in self.messages]
    
    def __str__(self):
        return str([str(message) for message in self.messages])

def log_retry_error(retry_state):
    logging.warning(f"Function failed after retries: {retry_state.outcome.exception()}")

@dataclass
class FunctionMetaInfo:
    module_name: str
    line_number: int
    stack_trace: list[str]

    def get_info_dict(self) -> dict:
        return {"module_name": self.module_name, "line_number": self.line_number, "stack_trace": str(self.stack_trace)}


@dataclass
class TokenStatistics:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def get_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }

    def __add__(self, target: TokenStatistics):
        return TokenStatistics(
            prompt_tokens=self.prompt_tokens + target.prompt_tokens,
            completion_tokens=self.completion_tokens + target.completion_tokens
        )
    
    def __lt__(self, target: TokenStatistics):
        return any((self.prompt_tokens < target.prompt_tokens, self.completion_tokens < target.completion_tokens))

@dataclass
class CompletionInfo:
    model: str
    raw_response: Any
    meta_info: FunctionMetaInfo
    token_statistics: TokenStatistics

    def get_info_dict(self) -> dict:
        return {"model": self.model, "token_statistics": self.token_statistics.get_dict(), "meta_info": self.meta_info.get_info_dict()}

@dataclass
class CompletionStatistics:
    get_response_func: str
    meta_info: FunctionMetaInfo
    token_statistics_dict: dict[str, TokenStatistics] = field(default_factory=dict)
    completion_info_list: List[CompletionInfo] = field(default_factory=list)

    def dump_token_statistics_dict(self):
        return {key: value.get_dict() for key, value in self.token_statistics_dict.items()}

    def get_overall_statistics_dict(self) -> dict:
        return {
            "get_response_func": self.get_response_func,
            "all_token_statistics": self.dump_token_statistics_dict(),
            "meta_info": self.meta_info.get_info_dict()
        }
    
    def get_detailed_statistics_dict(self) -> dict:
        return {
            "get_response_func": self.get_response_func,
            "all_token_statistics": self.dump_token_statistics_dict(),
            "all_completion_info": [completion_info.get_info_dict() for completion_info in self.completion_info_list],
            "meta_info": self.meta_info.get_info_dict()
        }

class TokenConsumptionExceededError(RuntimeError):
    pass

def _get_stack_info(frame: FrameType, max_depth: int = 10) -> list[str]:
    """
    回溯获取调用栈信息，返回格式化的字符串列表。
    列表顺序：从当前帧（最深层） -> 最外层。
    """
    stack_trace = []
    current_frame = frame
    depth = 0
    while current_frame and depth < max_depth:
        # 获取模块名：优先从 globals 取，取不到则标记为 unknown
        # 相比 inspect.getmodule，直接访问 f_globals 性能更好
        module_name = current_frame.f_globals.get('__name__', 'unknown')
        
        # 获取函数名
        func_name = current_frame.f_code.co_name
        
        # 获取行号
        line_number = current_frame.f_lineno
        
        # 格式化为字符串，例如: "my_module.my_func:42"
        stack_str = f"{module_name}.{func_name}:{line_number}"
        stack_trace.append(stack_str)
        # 指向上一层调用者
        current_frame = current_frame.f_back
        depth += 1
    return stack_trace

def _get_meta_info(frame: FrameType):
    import inspect
    module_name = inspect.getmodule(frame).__name__
    line_number = frame.f_lineno
    stack_trace = _get_stack_info(frame, max_depth=10)
    return FunctionMetaInfo(module_name=module_name, line_number=line_number, stack_trace=stack_trace)

def _parse_response_to_completion_info(raw_response, meta_info: FunctionMetaInfo):
    '''
    解析模块不能报错，避免给 get_response 带来麻烦。
    '''
    from openai.types.chat.chat_completion import ChatCompletion
    model = None
    token_statistics = None
    ctx_logger = retry_logger
    ctx_logger.bind(raw_response_type=type(raw_response))
    if isinstance(raw_response, ChatCompletion):
        model = raw_response.model
        token_statistics=TokenStatistics(
            prompt_tokens=raw_response.usage.prompt_tokens,
            completion_tokens=raw_response.usage.completion_tokens,
        )
    if isinstance(raw_response, (dict, list)):
        raise NotImplementedError("Json Format Response is Not Implemented!")
    else:
        try:
            import litellm
            if isinstance(raw_response, litellm.ModelResponse):
                model = raw_response.model
                token_statistics=TokenStatistics(
                    prompt_tokens=raw_response.usage.prompt_tokens,
                    completion_tokens=raw_response.usage.completion_tokens,
                )
        except Exception as e:
            ctx_logger.warn(f"Failed to import litellm to parse raw response", error=e)
    if model is None:
        model = "<unknown>"
        ctx_logger.warn(f"Failed to parse model from raw response, set to {model}")
    if token_statistics is None:
        token_statistics=TokenStatistics(
            prompt_tokens=0,
            completion_tokens=0,
        )
        ctx_logger.warn(f"Failed to parse token_statistics from raw response, set to {token_statistics}")
    return CompletionInfo(
        model=model,
        token_statistics=token_statistics,
        raw_response=raw_response,
        meta_info=meta_info
    )

def _add_completion_info(completion_statistics: CompletionStatistics, completion_info: CompletionInfo):
    completion_statistics.completion_info_list.append(completion_info)
    model = completion_info.model
    
    if completion_info.model not in completion_statistics.token_statistics_dict:
        completion_statistics.token_statistics_dict[model] = TokenStatistics()
    
    completion_statistics.token_statistics_dict[model] += completion_info.token_statistics

completion_statistics_dict: dict[Any, CompletionStatistics] = {}

def dump_completion_statistics_dict() -> dict:
    global completion_statistics_dict
    return [{"get_response_func": completion_statistics.get_response_func, "detailed": completion_statistics.get_detailed_statistics_dict()} for completion_statistics in completion_statistics_dict.values()]

def init_get_response_with_completion_statistics(get_response: Callable[[list], RichStr], init_response_args: InitResponseArgs=None):
    '''
    可以包装普通的 get_response，但是不能随意报错，一旦随意报错就麻烦了。
    添加了 budget 功能。
    '''
    import inspect
    global completion_statistics_dict
    token_budget = init_response_args and init_response_args.token_budget

    frame = inspect.currentframe()
    meta_info = _get_meta_info(frame.f_back)
    completion_statistics = CompletionStatistics(get_response_func=repr(get_response), meta_info=meta_info) # 默认而言，一个 get_response 对应一个 statistics。
    
    log_to_tensorboard = init_response_args.tensorboard_log_dir is not None
    writer = None
    if log_to_tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(log_dir=init_response_args.tensorboard_log_dir)
        atexit.register(writer.close) # 退出时自动关闭。
        
    def get_response_with_statistics(m):
        global completion_statistics_dict
        # 查询是否超出 budget，如果超出就停止
        if token_budget is not None:
            current_consumption = sum([completion_info.token_statistics for completion_info in completion_statistics_dict[get_response_with_statistics].completion_info_list], start=TokenStatistics())
            if current_consumption > token_budget:
                raise TokenConsumptionExceededError(f"You have exceeded the token consumption budget! This function has been banned! current consumption: {current_consumption}, token budget: {token_budget}")
        # 获取当前调用栈
        frame = inspect.currentframe()
        # 获取调用栈中的上一层（即调用 sample_function 的位置）
        meta_info = _get_meta_info(frame.f_back)
        response = get_response(m)
        try:
            # 记录相关信息。
            completion_info = _parse_response_to_completion_info(getattr(response, RichStr.RAW, None), meta_info)
            # 打印信息
            retry_logger.info(f"completion info: {completion_info.get_info_dict()}")
            
            _add_completion_info(completion_statistics, completion_info)

            # log to tensorboard
            if log_to_tensorboard:
                retry_logger.info(f"try to write log to tensorboard: {writer.logdir}")

                model_name = completion_info.model
                for key, value in completion_info.token_statistics.get_dict().items():
                    writer.add_scalar(f"step_token_usage/{model_name}/{key}/{meta_info.module_name}-{meta_info.line_number}", value, global_step=len(completion_statistics.completion_info_list))
                    writer.flush()
                for key, value in completion_statistics.token_statistics_dict[model_name].get_dict().items():
                    writer.add_scalar(f"total_token_usage/{model_name}/{key}/{meta_info.module_name}-{meta_info.line_number})", value, global_step=len(completion_statistics.completion_info_list))
                    writer.flush()
                retry_logger.info(f"succeeded to write log to tensorboard: {writer.logdir}")

        except Exception as e:
            retry_logger.error(f"Failed to parse completion info. This completion will not be recorded.", error=e, meta_info=meta_info, exc_info=True)
        return response
    
    completion_statistics_dict[get_response_with_statistics] = completion_statistics # 在全局字典中注册该信息

    return get_response_with_statistics

# local server model
def get_remote_response(prompt: str, server_url: str = "http://localhost:8888/generate"):
    payload = {"prompt": prompt}
    try:
        response = requests.post(server_url, json=payload)
        response.raise_for_status()  # 如果响应不是 200，会抛出异常
        data = response.json()
        return data["outputs"]
    except requests.RequestException as e:
        print(f"请求失败: {e}")
        return None

def get_remote_response_mm(prompt: str, images: List[Image.Image], server_url: str = "http://localhost:8888/generate_mm"):
    base64_images = []
    for image in images:
        base64_images.append(image_to_base64(image))
    payload = {"prompt": prompt, "base64_images": base64_images}
    try:
        response = requests.post(server_url, json=payload)
        response.raise_for_status()  # 如果响应不是 200，会抛出异常
        data = response.json()
        return data["outputs"]
    except requests.RequestException as e:
        print(f"请求失败: {e}")
        return None

@dataclass
class InitResponseArgs:
    model: str
    record_completion_statistics: bool = False
    token_budget: TokenStatistics = None
    tensorboard_log_dir: str = None
    base_url: str = None
    api_key: str = None
    completion_kwargs: dict = field(default_factory=dict)
    use_sdk: bool = True # 如果关闭，将使用 requests 库请求。

    def update_args(self, value: InitResponseArgs):
        for field in dataclasses.fields(self):
            if getattr(self, field.name) is None:
                new_value = getattr(value, field.name)
                retry_logger.info(f"field {field.name} is not set, fallback to default value ({new_value})")
                setattr(self, field.name, new_value)
        if not self.completion_kwargs:
            self.completion_kwargs.update(value.completion_kwargs)
        # for completion_key, completion_value in value.completion_kwargs.items():
        #     if completion_key not in self.completion_kwargs:
        #         retry_logger.info(f"completion kwargs {completion_key} is not set, fallback to default value ({completion_value})")
        #         self.completion_kwargs[completion_key] = completion_value


no_retry_get_response_error_types = (TokenConsumptionExceededError, KeyboardInterrupt, BdbQuit) # 遇到这些情况不 retry get response
retry_get_response_wrapper = tenacity.retry(stop=tenacity.stop_after_attempt(10), retry=tenacity.retry_if_not_exception_type(no_retry_get_response_error_types), wait=tenacity.wait_fixed(30), before_sleep=before_sleep_log(retry_logger, logging.WARNING), after=partial(log_retry_error_with_traceback, handler=lambda s: retry_logger.info(f"Attempt failed", error=s)))

def init_get_response(init_response_args: InitResponseArgs=None):
    if init_response_args.use_sdk:
        return init_get_litellm_response(init_response_args)
    else:
        return init_get_requests_response(init_response_args)

# 单例模式，只有唯一的 client!
openai_client = None
claude_client = None
local_openai_client = None # 本地 llm server

def init_get_local_openai_response(model="local model", server_url: str = None, init_response_args: InitResponseArgs=None):
    from openai import OpenAI, AzureOpenAI
    if init_response_args is not None:
        model = init_response_args.model
        record_completion_statistics = init_response_args.record_completion_statistics
        token_budget = init_response_args.token_budget
    global local_openai_client
    if local_openai_client is None:
        retry_logger.info(f"initializing client {OpenAI.__name__}")
        local_openai_client = OpenAI(
            base_url=server_url or os.environ['LOCAL_OPENAI_BASE_URL'],
            api_key=os.environ['LOCAL_OPENAI_API_KEY']
        )
        retry_logger.info(f"Successfully initialized client {type(local_openai_client)}")
    retry_logger.info(f"initializing get openai response", local_openai_client=local_openai_client, model=model)

    get_openai_response = lambda m: (lambda r: RichStr(r.choices[0].message.content, r))(local_openai_client.chat.completions.create(
        model=model,    # 通过 endpoint 指定模型
        messages=m,
        temperature=0.6,
        stream=False
    ))
    if record_completion_statistics:
        get_openai_response = init_get_response_with_completion_statistics(get_openai_response)
        
    return get_openai_response

def init_get_openai_response(model: str="gpt-4.1", use_azure=True, init_response_args: InitResponseArgs=None):
    '''
    可以选择 OpenAI, AzureOpenAI, 或者多个 Config 融合。
    '''
    from openai import OpenAI, AzureOpenAI
    if init_response_args is not None:
        model = init_response_args.model
        base_url = init_response_args.base_url
        api_key = init_response_args.api_key
    global openai_client
    if openai_client is None:
        # 若没有初始化，则初始化一下
        if use_azure:
            retry_logger.info(f"initializing client {AzureOpenAI.__name__}")
            openai_client = AzureOpenAI(
                azure_endpoint=base_url or os.environ['AZURE_OPENAI_ENDPOINT'],
                api_key=api_key or os.environ['AZURE_OPENAI_API_KEY'],
            )
        else:
            retry_logger.info(f"initializing client {OpenAI.__name__}")
            openai_client = OpenAI(
                base_url=base_url or os.environ['OPENAI_BASE_URL'],
                api_key=api_key or os.environ['OPENAI_API_KEY']
            )
        retry_logger.info(f"Successfully initialized client {type(openai_client)}")
    retry_logger.info(f"initializing get openai response", openai_client=openai_client, model=model)
    get_openai_response = lambda m: (lambda r: RichStr(r.choices[0].message.content, r))(openai_client.chat.completions.create(
        model=model,    # 通过 endpoint 指定模型
        messages=m,
        temperature=0.6,
        stream=False
    ))
    get_openai_response = retry_get_response_wrapper(get_openai_response)

    if init_response_args and init_response_args.record_completion_statistics:
        get_openai_response = init_get_response_with_completion_statistics(get_openai_response, init_response_args)
    return get_openai_response

def init_get_claude_response(model: str="claude-4-sonnet", init_response_args: InitResponseArgs=None):
    from openai import OpenAI, AzureOpenAI
    if init_response_args is not None:
        model = init_response_args.model
    global claude_client
    if claude_client is None:
        retry_logger.info(f"initializing client {OpenAI.__name__}")
        claude_client = OpenAI(
            base_url=os.environ['OPENAI_BASE_URL'],
            api_key=os.environ['OPENAI_API_KEY']
        )
        retry_logger.info(f"Successfully initialized client {type(claude_client)}")

    retry_logger.info(f"initializing get claude response", claude_client=claude_client, model=model)

    get_claude_response = lambda m: (lambda r: RichStr(r.choices[0].message.content, r))(claude_client.chat.completions.create(
        model=model,    # 通过 endpoint 指定模型
        messages=m,
        temperature=0.6,
        stream=False,
    ))
    get_claude_response = retry_get_response_wrapper(get_claude_response)
    if init_response_args and init_response_args.record_completion_statistics:
        get_claude_response = init_get_response_with_completion_statistics(get_claude_response, init_response_args)
    return get_claude_response

def init_get_litellm_response(init_response_args: InitResponseArgs=None):
    import litellm
    if init_response_args is not None:
        model = init_response_args.model
    retry_logger.info(f"initializing get litellm response", model=model)
    get_litellm_response = lambda m: (lambda r: (lambda _: RichStr(r.choices[0].message.content, r))(retry_logger.info(f"get response", response=r)))(litellm.completion(
        model=model,
        messages=m,
        base_url=init_response_args.base_url,
        api_key=init_response_args.api_key,
        **init_response_args.completion_kwargs
    ))
    get_litellm_response = retry_get_response_wrapper(get_litellm_response)
    if init_response_args and init_response_args.record_completion_statistics:
        get_litellm_response = init_get_response_with_completion_statistics(get_litellm_response, init_response_args)
    return get_litellm_response

def _handle_openai_requests(model_name: str, messages, base_url: str, api_key, **completion_kwargs):
    base_url = base_url.removesuffix('/')
    if not base_url.endswith('/chat/completions'):
        base_url = f"{base_url}/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "messages": messages
    }

    try:
        response = requests.post(base_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        response_data = response.json()
        return response_data
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"请求 OpenAI 时发生错误: {e}\n响应内容: {e.response.text if e.response else 'N/A'}")


def requests_completion(model: str, messages, base_url, api_key, **completion_kwargs):
    if "/" in model:
        provider, model_name = model.split("/", maxsplit=1)
    else:
        provider, model_name = "openai", model
    handler_dict = {
        "openai": _handle_openai_requests
    }

    return handler_dict[provider.lower()](model_name, messages, base_url, api_key, **completion_kwargs)

def init_get_requests_response(init_response_args: InitResponseArgs=None):
    """
    一个不依赖任何 SDK，仅使用 requests 调用大模型 API 的客户端。
    支持 OpenAI 请求格式。
    必须指定 API_KEY 和 BASE_URL。
    """
    if init_response_args is not None:
        model = init_response_args.model
    retry_logger.info(f"initializing get requests response", model=model)
    get_requests_response = lambda m: (lambda r: (lambda _: RichStr(r['choices'][0]['message']['content'], r))(retry_logger.info(f"get response", response=r)))(requests_completion(
        model=model,
        messages=m,
        base_url=init_response_args.base_url,
        api_key=init_response_args.api_key,
        **init_response_args.completion_kwargs
    ))
    get_requests_response = retry_get_response_wrapper(get_requests_response)
    if init_response_args and init_response_args.record_completion_statistics:
        get_requests_response = init_get_response_with_completion_statistics(get_requests_response, init_response_args)
    return get_requests_response


def init_get_parsed_response(get_response, response_parser, try_times=1, get_fix_response=None, prepare_fix_messages=None):
    get_fix_response = get_fix_response or get_response
    def default_get_fix_messages(messages: list, response_with_error: str) -> List[Dict[str, str]]:
        fix_messages = Messages(messages=[
            make_assistant(response_with_error),
            make_user("Your response does not meet the format I requested. Please respond according to the format I specified.")
        ]).serialize()
        return messages + fix_messages
        
    prepare_fix_messages = prepare_fix_messages or default_get_fix_messages
    def get_parsed_response(messages):
        '''
        请求并解析。如果解析失败，要求模型修复问题。
        get_fix_response: 可以指定修复问题的模型，若不指定，默认使用原来的模型修复。
        '''
        ctx_logger = retry_logger
        res = None
        for attempt in tenacity.Retrying(stop=tenacity.stop_after_attempt(try_times), retry=tenacity.retry_if_not_exception_type(no_retry_get_response_error_types), before=lambda s: ctx_logger.info(f"Attempt {s.attempt_number} to request parsed response"), after=partial(log_retry_error_with_traceback, handler=lambda s: ctx_logger.info(f"Attempt failed", error=s))):
            with attempt:
                request_messages = messages
                if not isinstance(res, str): # 说明 res 变成了奇怪的东西，强制改为 None。
                    res = None
                if res is not None: # 这说明上一次已经获得了可以正确解析的 response
                    request_messages = prepare_fix_messages(messages, res)
                ctx_logger = ctx_logger.bind(messages=(lambda x: x[:300] + f"...({len(x) - 600} more chars)..." + x[-300:] if len(x) > 600 else x)(str(request_messages)), messages_len=len(request_messages))
                ctx_logger.info("try to get response")
                res = (get_response if attempt.retry_state.attempt_number == 1 else get_fix_response)(request_messages)
                ctx_logger = ctx_logger.bind(response=res)
                ctx_logger.info("got response")
                parsed_res = response_parser(res)
                ctx_logger = ctx_logger.bind(response=parsed_res)
                ctx_logger.info("got parsed response")
                return parsed_res, res
    return get_parsed_response