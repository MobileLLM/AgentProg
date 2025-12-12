from dataclasses import dataclass
import structlog
import logging
from itertools import chain
logger = logging.getLogger()
LOG_LEVEL = logging.WARN # 全局日志级别

def enable_log(logging_path: str = None):
    def filter_by_level(
        logger: logging.Logger, method_name: str, event_dict
    ):
        if logger.isEnabledFor(structlog.stdlib.NAME_TO_LEVEL[method_name]):
            return event_dict

        raise structlog.stdlib.DropEvent

    def remove_meta_in_console_renderer(_, __, event_dict: structlog.typing.EventDict):
        event_dict['location'] = f"[{event_dict['module']}:{event_dict['lineno']}]"
        event_dict['func_name'] = f"[{event_dict['func_name']}]"
        for key in ["module", "lineno", "_record", "_from_structlog"]:
            if key in event_dict:
                del event_dict[key]
        return event_dict

    plus_processor = structlog.processors.CallsiteParameterAdder(
        parameters=[
            structlog.processors.CallsiteParameter.FUNC_NAME,
            structlog.processors.CallsiteParameter.MODULE,
            structlog.processors.CallsiteParameter.LINENO,
        ]
    )
    shared_processors = [structlog.stdlib.add_logger_name, filter_by_level, plus_processor, remove_meta_in_console_renderer, *structlog.get_config()['processors']]
    structlog.configure(
        logger_factory=structlog.stdlib.LoggerFactory(),
        processors=shared_processors,
        cache_logger_on_first_use=True
    )
    handlers = []
    stream_handler = logging.StreamHandler()
    stream_formatter = logging.Formatter('%(message)s')
    stream_handler.setFormatter(stream_formatter)
    handlers.append(stream_handler)
    if logging_path is not None:
        print(f"write logs to logging path: {logging_path}")
        file_handler = logging.FileHandler(logging_path, encoding='utf-8')
        file_formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
    # logging.basicConfig(handlers=handlers, level=LOG_LEVEL)

def get_filehandler(logging_path):
    file_handler = logging.FileHandler(logging_path, encoding='utf-8')
    file_formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(file_formatter)
    return file_handler

@dataclass
class LogConfig:
    _logging_path: str = None

    @property
    def logging_path(self):
        return self._logging_path
    
    @logging_path.setter
    def logging_path(self, value):
        if value == self._logging_path: # 没有修改，跳过
            return
        self._logging_path = value
        if value is None:
            return

        for key, logger in logger_dict.items():
            file_handler = get_filehandler(value)
            
            for handler in logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    logger.info(f"remove old handler: {handler}")
                    logger.removeHandler(handler)
                    handler.close()

            logger.addHandler(file_handler)
            logger.info(f"update logging path: {value}")

global_log_config = LogConfig()
logger_dict: dict[str, structlog.stdlib.BoundLogger] = {}

def get_logger(name):
    if name in logger_dict:
        return logger
    
    log_config = global_log_config
    
    def filter_by_level(
        logger: logging.Logger, method_name: str, event_dict
    ):
        if logger.isEnabledFor(structlog.stdlib.NAME_TO_LEVEL[method_name]):
            return event_dict

        raise structlog.stdlib.DropEvent

    def remove_meta_in_console_renderer(_, __, event_dict: structlog.typing.EventDict):
        event_dict['location'] = f"[{event_dict['module']}:{event_dict['lineno']}]"
        event_dict['func_name'] = f"[{event_dict['func_name']}]"
        for key in ["module", "lineno", "_record", "_from_structlog"]:
            if key in event_dict:
                del event_dict[key]
        return event_dict

    plus_processor = structlog.processors.CallsiteParameterAdder(
        parameters=[
            structlog.processors.CallsiteParameter.FUNC_NAME,
            structlog.processors.CallsiteParameter.MODULE,
            structlog.processors.CallsiteParameter.LINENO,
        ]
    )
    raw_logger = logging.getLogger(name)
    raw_logger.setLevel(LOG_LEVEL)
    
    raw_logger.propagate = False 
    raw_logger.handlers.clear()
    
    my_processors = [
        structlog.stdlib.add_logger_name, filter_by_level, plus_processor, remove_meta_in_console_renderer, *structlog.get_config()['processors'],
    ]
    
    logger: structlog.stdlib.BoundLogger = structlog.wrap_logger(
        raw_logger,
        processors=my_processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True
    )

    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(message)s') 
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logging_path = log_config.logging_path
    if logging_path is not None:
        file_handler = get_filehandler(logging_path)
        logger.addHandler(file_handler)
        logger.info(f"logging path: {logging_path}")
    
    logger_dict[name] = logger

    return logger

if __name__ != "__main__":
    from agentprog.all_utils import log_utils
    module_logger = log_utils.get_logger(__name__)
    context_logger = module_logger.bind(log_level=logging.getLevelName(LOG_LEVEL))
    context_logger.info("Initialized logging basic config")