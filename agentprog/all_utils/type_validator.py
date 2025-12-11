'''
### Functionality Description

This document describes the implementation of a type validator with the following supported syntax and features:

**I. Type Validation**

1.  **Basic Types:**
    Supports `bool`, `int`, `float`, `str`, and constants (e.g., `"test"`).

2.  **Union of Basic Types:**
    Supports union type declarations like `bool | int`.

3.  **Homogeneous List Types:**
    e.g., `list[str]` represents a list where all elements are of type `str`.

4.  **Homogeneous Key-Value Dictionary Types:**
    e.g., `dict[str, int]` represents a dictionary where all keys are of type `str` and all values are of type `int`.

5.  **Nested Lists (Detailed Control):**
    For more precise control over list structures, `list` can be replaced with `[<type spec>]`. The nested element type `<type spec>` can be any type conforming to the rules in this document.

    5.1. **Fixed-Length Lists:**
    e.g., `[str, int, int]` represents a list of length 3, where the first element is a `str`, the second is an `int`, and the third is an `int`.

    5.2. **Variable-Length Lists:**
    e.g., `[int, str, ...]` represents a list of any length where each element can be either an `int` or a `str`.
    `[int | str, ...]` can also express the same meaning.

6.  **Nested Dictionaries (Detailed Control):**
    For more precise control over dictionary formats, `dict` can be replaced with `{<type spec1>: <type spec2>}`. The nested key type `<type spec1>` and value type `<type spec2>` can be any type conforming to the rules in this document (provided it's supported by Python; for example, `{[1]: 1}` would raise an error in Python).

    6.1. **Fixed-Structure Dictionaries:**
    e.g., `{"math": int, "english": int, str: list[bool]}` represents a dictionary that must have key-value pairs matching these types: constant keys `"math"` and `"english"` with `int` values, and a `str` key with a `list[bool]` value.
    *(Note: Since dictionaries are unordered, validating a dictionary against a fixed-structure type requires finding a perfect match between the dictionary's items and the type's key-value specifications. The Hungarian algorithm can be used to find such a perfect matching, with a complexity of O(n^3) for matching n keys to n locks.)*

    6.2. **Variable-Structure Dictionaries:** (This feature is considered less practical and is suggested to be removed.)
    e.g., `{str: int | str, ...:...}` represents a dictionary of any length where each key-value pair must be either `(str, int)` or `(str, str)`.

7.  **Annotations (Type Descriptions):**
    You can add descriptive annotations to any type defined in this document. The syntax is to replace a type `<type spec>` with `(<description>, <type spec>)`. For example, `("age", int)` is equivalent to `int`; `{("subject", str): ("score", int), ...:...}` is equivalent to `{str: int, ...:...}`. Annotations can be nested and placed wherever appropriate to provide context, which can be supplied to large language models.

**II. Type Explanation and Examples**

This feature translates the type syntax described above into a natural language format that a large language model can understand, incorporating any user-written annotations. It also generates a sample data example for the given type.

**III. Error Messages**

If the data does not conform to the type requirements, the error message should clearly indicate which part of the test data fails to meet which specific requirement of the type specification.

'''

from copy import deepcopy
import json
from types import UnionType, NoneType
from typing import get_args, get_origin
from dataclasses import dataclass, field
from typing import List, Any, Optional
import traceback
from agentprog.all_utils.log_utils import structlog

from agentprog.all_utils import log_utils
logger = log_utils.get_logger(__name__)
type_names = {str: "字符串", int: "整数", float: "浮点数", bool: "布尔值"}
basic_types = (int, float, str, bool, NoneType)
basic_values = (None,)

@dataclass
class ConvertResult:
    origin_data: Any
    is_converted: bool
    converted_data: Any
    path: str = None

def _try_to_convert_basic_type(data, target_type):
    '''
    目前仅支持从 str -> int, bool, float.
    '''
    logger.warn(f"Trying to convert data from type {type(data)} to type {target_type}", data=data)
    try:
        if not isinstance(data, str):
            raise ValueError(f"type {type(data)} to {target_type} is not supported.")
        if target_type in (int, float):
            converted_data = target_type(data)
            logger.warn(f"Convert data from type {type(data)} to type {target_type}", data=data, converted_data=converted_data)
            return ConvertResult(data, True, converted_data)
        if target_type in (bool,):
            def _str_to_bool(s):
                if s.lower() == 'false':
                    return False
                elif s.lower() == 'true':
                    return True
                else:
                    raise ValueError("Invalid string to convert to bool")
            converted_data = _str_to_bool(data)
            logger.warn(f"Convert data from type {type(data)} to type {target_type}", data=data, converted_data=converted_data)
            return ConvertResult(data, True, converted_data)
        else:
            logger.error(f"Failed to convert data from type {type(data)} to type {target_type}, because of unmatched types", data=data)
            return ConvertResult(data, False, None)
    except Exception as e:
        logger.error(f"Failed to convert data from type {type(data)} to type {target_type}, error: {e}", data=data)
        return ConvertResult(data, False, None)

def hungarian_algorithm(keys, locks, can_open):
    """
    完美匹配算法。
    keys: 钥匙列表
    locks: 锁列表  
    can_open: can_open(i, j) = True 表示钥匙i能开锁j
    """
    n = len(keys)
    match_lock = [-1] * n  # match_lock[j] = i 表示锁j被钥匙i匹配
    
    def dfs(key, visited):
        for lock in range(n):
            if can_open(keys[key], locks[lock]) and not visited[lock]:
                visited[lock] = True
                # 如果锁未被匹配，或者能找到增广路径
                if match_lock[lock] == -1 or dfs(match_lock[lock], visited):
                    match_lock[lock] = key
                    return True
        return False
    
    matched = 0
    for key in range(n):
        visited = [False] * n
        if dfs(key, visited):
            matched += 1
    
    if matched == n:
        # 构造匹配结果
        result = {}
        for lock in range(n):
            if match_lock[lock] != -1:
                result[keys[match_lock[lock]]] = locks[lock]
        return result
    else:
        return None  # 无完美匹配

def generate_example(type_spec, max_list_length=3, max_dict_items=2):
    '''
    根据类型规范生成一个合法的示例值
    
    Args:
        type_spec: 类型规范
        max_list_length: 变长列表的最大长度
        max_dict_items: 变长字典的最大键值对数量
    '''
    import random
    try:
        if isinstance(type_spec, tuple):
            description, target_type = type_spec
        else:
            description, target_type = None, type_spec
    except:
        logger.error("parse type spec failed!", type_spec=str(type_spec))
        raise
    desc_suffix = f" of {description}" if description is not None else ""
    # 基础类型
    if isinstance(target_type, basic_types) or target_type in basic_values:  # 常量值
        return target_type
    elif target_type == str:
        return f"<string{desc_suffix}>"
    elif target_type == int:
        return f"<int{desc_suffix}>"
    elif target_type == float:
        return f"<float{desc_suffix}>"
    elif target_type == bool:
        return f"<bool{desc_suffix}>"
    elif target_type == NoneType:
        return None
    
    # 列表类型
    elif isinstance(target_type, list):
        if len(target_type) == 0:
            return []
        elif target_type[-1] == ...:  # 变长列表
            # 随机选择一个类型，生成1-max_list_length个元素
            element_types = target_type[:-1]
            length = random.randint(2, max_list_length)
            result = []
            for _ in range(length):
                chosen_type = random.choice(element_types)
                result.append(generate_example(chosen_type, max_list_length, max_dict_items))
            return result
        else:  # 定长列表
            result = []
            for single_type in target_type:
                result.append(generate_example(single_type, max_list_length, max_dict_items))
            return result
    
    elif target_type == list:
        return []
    
    elif get_origin(target_type) == list:
        element_type = get_args(target_type)
        if len(element_type) == 0:
            return []
        # 生成1-max_list_length个元素
        length = random.randint(2, max_list_length)
        result = []
        for _ in range(length):
            result.append(generate_example(element_type[0], max_list_length, max_dict_items))
        return result

    # 字典类型
    elif isinstance(target_type, dict):
        if len(target_type) == 0:
            return {}
        elif ... in target_type and target_type[...] == ...:  # 变长字典
            # 生成1-max_dict_items个键值对
            result = {}
            key_value_pairs = [(k, v) for k, v in target_type.items() if k != ... and v != ...]
            if key_value_pairs:
                chosen_pair = random.choice(key_value_pairs)
                key_type, value_type = chosen_pair
                # 生成多个键值对
                for i in range(random.randint(2, max_dict_items)):
                    key = generate_example(key_type, max_list_length, max_dict_items)
                    # 如果key是字符串，为了避免重复，添加序号
                    if isinstance(key, str):
                        key = f"{key}({i+1})"
                    value = generate_example(value_type, max_list_length, max_dict_items)
                    result[key] = value
            return result
        else:  # 定长字典
            result = {}
            for key_type, value_type in target_type.items():
                key = generate_example(key_type, max_list_length, max_dict_items)
                value = generate_example(value_type, max_list_length, max_dict_items)
                result[key] = value
            return result
    
    elif target_type == dict:
        return {}
    
    elif get_origin(target_type) == dict:
        key_value_types = get_args(target_type)
        if not key_value_types:
            return {}
        key_type, value_type = key_value_types
        # 生成1-max_dict_items个键值对
        result = {}
        for i in range(random.randint(1, max_dict_items)):
            key = generate_example(key_type, max_list_length, max_dict_items)
            if isinstance(key, str):
                key = f"{key}_{i+1}"
            value = generate_example(value_type, max_list_length, max_dict_items)
            result[key] = value
        return result

    # Union类型
    elif get_origin(target_type) == UnionType:
        # 随机选择一个类型
        union_types = get_args(target_type)
        chosen_type = random.choice(union_types)
        return generate_example(chosen_type, max_list_length, max_dict_items)
    
    else:
        raise ValueError(f"Unexpected type of spec: {repr(type_spec)}")

def generate_json_example(type_spec, max_list_length=3, max_dict_items=2):
    '''
    生成 json 的 example，generate_example 只生成一个 python 的对象，我们要将其变成 json 格式。
    '''
    example = generate_example(type_spec=type_spec, max_list_length=max_list_length, max_dict_items=max_dict_items)
    json_example = json.dumps(example, ensure_ascii=False)
    import re

    # 去掉不必需要的引号
    pattern = r'^"<(string|int|float|bool)([ a-zA-Z_]*)>"$'
    json_example = re.sub(pattern, r'<\1\2>', json_example)
    return json_example

def explain_type(type_spec, indent_level=0, parent_index=""):
    '''
    将类型转换为自然语言格式说明，支持缩进和编号。
    
    Args:
        type_spec: 类型规范
        indent_level: 当前缩进级别
        parent_index: 父级编号前缀
    '''
    indent = "  " * indent_level  # 每级缩进2个空格
    
    if isinstance(type_spec, tuple):
        description, target_type = type_spec
    else:
        description, target_type = None, type_spec
    
    # 类型名称映射
    type_names = {
        str: "字符串",
        int: "整数",
        float: "浮点数", 
        bool: "布尔值",
        list: "列表",
        dict: "字典",
        NoneType: "null"
    }

    # 构建描述后缀
    desc_suffix = f"，表示：{description}" if description else ""
    
    # 基础类型
    if isinstance(target_type, basic_types) or target_type in basic_values:  # value type
        if target_type is None or target_type is NoneType:
            constant_value = "null"
        else:
            constant_value = repr(target_type)
        return f"{indent}{parent_index}为{type_names[type(target_type)]}常量 {constant_value}{desc_suffix}"
    elif target_type in basic_types:  # other basic type
        return f"{indent}{parent_index}为{type_names[target_type]}类型{desc_suffix}"
    
    # 列表类型 list[TupleSpec]
    elif isinstance(target_type, list):
        if len(target_type) == 0:
            return f"{indent}{parent_index}为列表类型{desc_suffix}"
        elif target_type[-1] == ...:  # 变长列表
            child_explains = []
            for i, single_type in enumerate(target_type[:-1], 1):
                child_index = f"{parent_index}{i}." if parent_index else f"{i}."
                child_explains.append(explain_type(single_type, indent_level + 1, child_index))
            
            child_explain = "\n".join(child_explains)
            
            if len(child_explains) == 1:  # 只有一种类型
                return f"{indent}{parent_index}为列表类型{desc_suffix}，列表的每一个元素必须符合下列类型要求：\n{child_explain}"
            else:
                return f"{indent}{parent_index}为列表类型{desc_suffix}，列表的每一个元素必须符合下列类型要求中的一种：\n{child_explain}"
        else:  # 定长的列表
            child_explains = []
            for i, single_type in enumerate(target_type, 1):
                child_index = f"{parent_index}{i}." if parent_index else f"{i}."
                child_explains.append(explain_type(single_type, indent_level + 1, child_index))
            
            child_explain = "\n".join(child_explains)
            return f"{indent}{parent_index}为列表类型{desc_suffix}，列表具有{len(target_type)}个元素，列表的元素分别为：\n{child_explain}"
    
    elif target_type == list:
        return f"{indent}{parent_index}为列表类型{desc_suffix}"

    elif get_origin(target_type) == list:  # is list[<some type>]
        element_type = get_args(target_type)
        if len(element_type) == 0:
            return f"{indent}{parent_index}为列表类型{desc_suffix}"
        
        child_index = f"{parent_index}1." if parent_index else "1."
        element_explain = explain_type(element_type[0], indent_level + 1, child_index)
        return f"{indent}{parent_index}为列表类型{desc_suffix}，列表的元素类型为：\n{element_explain}"

    # 字典类型 Dict[TupleSpec, TupleSpec]
    elif isinstance(target_type, dict):
        if len(target_type) == 0:
            return f"{indent}{parent_index}为字典类型{desc_suffix}"

        elif ... in target_type and target_type[...] == ...:  # 变长字典
            child_explains = []
            i = 1
            for key_type, value_type in target_type.items():
                if key_type != ... and value_type != ...:
                    child_index = f"{parent_index}{i}." if parent_index else f"{i}."
                    key_explain = explain_type(key_type, indent_level + 2, f"{child_index}")
                    value_explain = explain_type(value_type, indent_level + 2, f"{child_index}")
                    insert_after_leading_space = lambda t, s: s[:len(s) - len(s.lstrip())] + t + s.lstrip()
                    key_explain = insert_after_leading_space("键：", key_explain)
                    value_explain = insert_after_leading_space("值：", value_explain)
                    child_explains.append(f"{indent}  {child_index}键值对类型：\n{key_explain}\n{value_explain}")
                    i += 1

            child_explain = "\n".join(child_explains)

            if len(child_explains) == 1:
                return f"{indent}{parent_index}为字典类型{desc_suffix}，键值对数量不限，字典的键值对类型必须符合下面的类型要求：\n{child_explain}"
            else:
                return f"{indent}{parent_index}为字典类型{desc_suffix}，键值对数量不限，字典的键值对类型必须符合下面的类型要求中的一种：\n{child_explain}"

        else:  # 定长的字典
            child_explains = []
            for i, (key_type, value_type) in enumerate(target_type.items(), 1):
                child_index = f"{parent_index}{i}." if parent_index else f"{i}."
                key_explain = explain_type(key_type, indent_level + 2, f"{child_index}")
                value_explain = explain_type(value_type, indent_level + 2, f"{child_index}")
                insert_after_leading_space = lambda t, s: s[:len(s) - len(s.lstrip())] + t + s.lstrip()
                key_explain = insert_after_leading_space("键：", key_explain)
                value_explain = insert_after_leading_space("值：", value_explain)
                child_explains.append(f"{indent}  {child_index}第{i}个键值对\n{key_explain}\n{value_explain}")
            
            child_explain = "\n".join(child_explains)
            return f"{indent}{parent_index}为字典类型{desc_suffix}，字典具有{len(target_type)}个键值对，依次满足以下类型要求：\n{child_explain}"
    
    elif target_type == dict:
        return f"{indent}{parent_index}为字典类型{desc_suffix}"

    elif get_origin(target_type) == dict:
        key_value_types = get_args(target_type)
        if not key_value_types:
            return f"{indent}{parent_index}为字典类型{desc_suffix}"
        
        key_type, value_type = key_value_types
        key_index = f"{parent_index}键: " if parent_index else "键: "
        value_index = f"{parent_index}值: " if parent_index else "值: "
        
        key_explain = explain_type(key_type, indent_level + 1, key_index)
        value_explain = explain_type(value_type, indent_level + 1, value_index)
        return f"{indent}{parent_index}为字典类型{desc_suffix}，字典的键值对类型为：\n{key_explain}\n{value_explain}"

    # 和类型的支持
    elif get_origin(target_type) == UnionType:
        child_explains = []
        for i, single_type in enumerate(get_args(target_type), 1):
            child_index = f"{parent_index}{i}." if parent_index else f"{i}."
            child_explains.append(explain_type(single_type, indent_level + 1, child_index))
        
        union_explain = "\n".join(child_explains)
        return f"{indent}{parent_index}符合下列类型要求中的任意一种{desc_suffix}：\n{union_explain}"

    else:
        raise ValueError(f"Unexpected type of spec: {repr(type_spec)}")

# 使用示例
def explain_with_example(type_spec, indent_level=0, parent_index=""):
    '''
    生成类型说明和示例
    '''
    explanation = explain_type(type_spec, indent_level, parent_index)
    try:
        example = generate_example(type_spec)
        return f"{explanation}\n\n示例值：{repr(example)}\n\n"
    except Exception as e:
        return f"{explanation}\n\n示例生成失败：{str(e)}\n\n"

def validate_type(data, type_spec):
    """递归验证类型，包含基本类型，常量，列表和字典"""
    '''
    规则：
    若使用实例：
    tuple 的长度永远为 2，解包为 description, type，仅验证 type，表达单项。type 也可以是 value.
    list 解包为 tuple or type, 可以加省略号，表达列表
    dict 解包为 tuple or type: tuple or type, 可以加省略号，表达字典
    
    若使用泛型：
    list 仅支持同质列表
    dict 仅支持同质键和同质值
    '''
    if isinstance(type_spec, tuple):
        description, target_type = type_spec
    else:
        description, target_type = None, type_spec
    # 基础类型
    if isinstance(target_type, basic_types): # value type
        return data == target_type
    elif target_type in basic_types: # other basic type
        return isinstance(data, target_type)
    
    # 列表类型 list[TupleSpec]
    elif isinstance(target_type, list):
        # 验证 list 内部是否符合规范
        if not isinstance(data, list):
            return False
        if len(target_type) == 0:
            return True
        elif target_type[-1] == ...: # 变长列表，same type as before
            return all(any(validate_type(element, single_type) for single_type in target_type[:-1]) for element in data)
        else: # 定长的列表
            if len(data) == len(target_type):
                return all(validate_type(element, single_type) for element, single_type in zip(data, target_type))
            else:
                return False
    
    elif target_type == list:
        return isinstance(data, list)

    elif get_origin(target_type) == list: # is list[<some type>]，仅支持同质列表。
        if not isinstance(data, list):
            return False
        
        element_type = get_args(target_type)
        if len(element_type) == 0:
            return True
        return all(validate_type(element, element_type[0]) for element in data)

    # 字典类型 Dict[TupleSpec, TupleSpec]
    elif isinstance(target_type, dict):
        if not isinstance(data, dict):
            return False
        if len(target_type) == 0:
            return True
        elif ... in target_type and target_type[...] == ...: # 变长字典，same type as before
            return all(any(validate_type(key, key_type) and validate_type(value, value_type) for key_type, value_type in target_type.items()) for key, value in data.items())
        else: # 定长的字典，匹配字典中的每一个键值对是否满足类型的要求。但是键值对是无序的，这一点要注意，如何贪婪匹配？只要其中一种模式满足即可。其实这个就是二分图的完美匹配问题，用匈牙利算法可解，复杂度 O(n^3)
            if len(data) == len(target_type):
                data_items = list(data.items())
                type_items = list(target_type.items())
                
                data_idxs = list(range(len(data_items)))
                type_idxs = list(range(len(type_items)))

                def match_function(data_idx, type_idx):
                    key_errors = validate_type(data_items[data_idx][0], type_items[type_idx][0], "")
                    value_errors = validate_type(data_items[type_idx][1], type_items[type_idx][1], "")
                    return not key_errors and not value_errors
                
                match_result = hungarian_algorithm(data_idxs, type_idxs, match_function)
                # match_result = hungarian_algorithm(list(data.items()), list(target_type.items()), lambda data_item, type_item: validate_type(data_item[0], type_item[0]) and validate_type(data_item[1], type_item[1]))
                if match_result is not None:
                    return True
                else:
                    return False
            else:
                return False
    
    elif target_type == dict:
        return isinstance(data, dict)

    # 和类型的支持
    elif get_origin(target_type) == UnionType:
        return any(validate_type(data, single_type) for single_type in get_args(target_type))

    elif get_origin(target_type) == dict:
        # 匹配字典的类型，我们默认字典的键类型是一致的，值类型是一致的。
        if not isinstance(data, dict):
            return False
        key_value_types = get_args(target_type)
        if not key_value_types:
            return True
        key_type, value_type = key_value_types
        return all(validate_type(key, key_type) and validate_type(value, value_type) for key, value in data.items())

    else:
        raise ValueError(f"Unexpected type of spec: {repr(type_spec)}")


@dataclass
class ValidationError:
    """单个验证错误的详细信息"""
    path: str                    # 错误位置路径
    message: str                 # 错误描述信息
    expected_type: Any           # 期望的类型
    actual_value: Any            # 实际的值
    actual_type: type            # 实际的类型
    description: Optional[str] = None  # 类型描述（如果有的话）
    
    def __str__(self):
        return self.message
    
    def to_dict(self):
        """转换为字典格式，便于序列化"""
        return {
            "path": self.path,
            "message": self.message,
            "expected_type": str(self.expected_type),
            "actual_value": repr(self.actual_value),
            "actual_type": self.actual_type.__name__,
            "description": self.description
        }

@dataclass
class ValidationResult:
    """验证结果对象"""
    is_valid: bool                      # 是否验证通过
    errors: List[ValidationError]       # 错误列表
    data: Any                          # 被验证的原始数据
    type_spec: Any                     # 类型规范
    is_converted: bool
    convert_results: List[ConvertResult]
    converted_data: Any = None
    @property
    def error_count(self) -> int:
        """错误数量"""
        return len(self.errors)
    
    @property
    def success(self) -> bool:
        """是否成功（is_valid的别名）"""
        return self.is_valid
    
    def get_errors_by_path(self, path_prefix: str) -> List[ValidationError]:
        """获取指定路径前缀的所有错误"""
        return [error for error in self.errors if error.path.startswith(path_prefix)]
    
    def get_first_error(self) -> Optional[ValidationError]:
        """获取第一个错误"""
        return self.errors[0] if self.errors else None
    
    def get_error_summary(self) -> str:
        """获取错误摘要"""
        if self.is_valid:
            return "验证通过"
        
        if self.error_count == 1:
            return f"发现1个错误: {self.errors[0].message}"
        else:
            return f"发现{self.error_count}个错误"
    
    def get_detailed_report(self) -> str:
        """获取详细的错误报告"""
        if self.is_valid:
            return "数据类型验证通过"
        
        lines = [f"数据类型验证失败，发现 {self.error_count} 个错误：\n"]
        
        for i, error in enumerate(self.errors, 1):
            lines.append(f"{i}. {error.message}")
            if error.description:
                lines.append(f"   类型描述: {error.description}")
            lines.append(f"   错误路径: {error.path}")
            lines.append(f"   期望类型: {explain_type(error.expected_type)}")
            lines.append(f"   实际值: {repr(error.actual_value)} ({repr(error.actual_type)})")
            lines.append("")
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "is_valid": self.is_valid,
            "error_count": self.error_count,
            "errors": [error.to_dict() for error in self.errors],
            "summary": self.get_error_summary()
        }
    
    def __bool__(self):
        """支持布尔值判断"""
        return self.is_valid
    
    def __str__(self):
        return self.get_error_summary()

@dataclass
class RecursiveValidationInfo:
    validation_errors: List[ValidationError] = field(default_factory=list)
    convert_results: List[ConvertResult] = field(default_factory=list)

    @property
    def is_valid(self):
        return len(self.validation_errors) == 0

    @property
    def need_convert(self):
        return len(self.convert_results) > 0

    def get_converted_data(self, original_data):
        '''
        获取转换后的数据.
        不允许修改 original_data 的值。
        '''
        if self.need_convert:
            return self.convert_results[0].converted_data
        else:
            return None

@dataclass
class RecursiveValidateArgs:
    data: Any
    type_spec: Any
    target_type: Any
    path: str
    description: str

def _validate_value_types(recursive_validate_args: RecursiveValidateArgs) -> RecursiveValidationInfo:
    data = recursive_validate_args.data
    target_type = recursive_validate_args.target_type
    path = recursive_validate_args.path
    description = recursive_validate_args.description
    recursive_validation_info = RecursiveValidationInfo()
    
    if data != target_type:
        recursive_validation_info.validation_errors.append(ValidationError(
            path=path,
            message=f"位置 {path} 的值应为常量 {repr(target_type)}，但实际值为 {repr(data)}",
            expected_type=target_type,
            actual_value=data,
            actual_type=type(data),
            description=description
        ))
    return recursive_validation_info

def _validate_basic_types(recursive_validate_args: RecursiveValidateArgs):
    data = recursive_validate_args.data
    target_type = recursive_validate_args.target_type
    path = recursive_validate_args.path
    description = recursive_validate_args.description
    recursive_validation_info = RecursiveValidationInfo()
    if not isinstance(data, target_type): 
        error_flag = False
        if path == 'data': # 目前只有当 path 为最浅的一层时生效。
            # try to convert for the basic types
            convert_result = _try_to_convert_basic_type(data, target_type)
            if convert_result.is_converted:
                recursive_validation_info.convert_results.append(convert_result)
            else:
                error_flag = True
        else:
            error_flag = True
        
        if error_flag: # convert failed, record error
            expected_name = type_names.get(target_type, str(target_type))
            actual_name = type_names.get(type(data), type(data).__name__)
            recursive_validation_info.validation_errors.append(ValidationError(
                path=path,
                message=f"位置 {path} 应为{expected_name}类型，但实际为{actual_name}类型，值为 {repr(data)}",
                expected_type=target_type,
                actual_value=data,
                actual_type=type(data),
                description=description
            ))
    return recursive_validation_info

def _validate_detailed_list_type(recursive_validate_args: RecursiveValidateArgs):
    data = recursive_validate_args.data
    target_type = recursive_validate_args.target_type
    path = recursive_validate_args.path
    description = recursive_validate_args.description
    recursive_validation_info = RecursiveValidationInfo()
    if not isinstance(data, list):
        recursive_validation_info.validation_errors.append(ValidationError(
            path=path,
            message=f"位置 {path} 应为列表类型，但实际为{type(data).__name__}类型，值为 {repr(data)}",
            expected_type=list,
            actual_value=data,
            actual_type=type(data),
            description=description
        ))
        return recursive_validation_info
    
    if len(target_type) == 0:
        return recursive_validation_info
    
    elif target_type[-1] == ...:  # 变长列表
        element_types = target_type[:-1]
        for i, element in enumerate(data):
            element_path = f"{path}[{i}]"
            # 检查是否匹配任一类型
            matched = False
            all_type_errors = []
            
            for j, single_type in enumerate(element_types):
                element_validate_recursive_info = _validate_recursive(element, single_type, element_path)
                if not element_validate_recursive_info.validation_errors:  # 没有错误，匹配成功
                    matched = True
                    break
                else:
                    all_type_errors.extend(element_validate_recursive_info.validation_errors)
            
            if not matched:
                # 创建一个汇总错误
                if len(element_types) == 1:
                    message = f"位置 {element_path} 不符合列表元素的类型要求"
                else:
                    message = f"位置 {element_path} 不符合列表元素的任何类型要求"
                
                recursive_validation_info.validation_errors.append(ValidationError(
                    path=element_path,
                    message=message + f"，详细错误: {'; '.join([e.message for e in all_type_errors[:3]])}{'...' if len(all_type_errors) > 3 else ''}",
                    expected_type=element_types[0] if len(element_types) == 1 else element_types,
                    actual_value=element,
                    actual_type=type(element),
                    description=description
                ))
    
    else:  # 定长列表
        if len(data) != len(target_type):
            recursive_validation_info.validation_errors.append(ValidationError(
                path=path,
                message=f"位置 {path} 的列表长度应为 {len(target_type)}，但实际长度为 {len(data)}",
                expected_type=f"长度为{len(target_type)}的列表",
                actual_value=data,
                actual_type=type(data),
                description=description
            ))
        
        # 继续验证每个元素
        min_len = min(len(data), len(target_type))
        for i in range(min_len):
            element_path = f"{path}[{i}]"
            element_recursive_validation_info = _validate_recursive(data[i], target_type[i], element_path)
            recursive_validation_info.validation_errors.extend(element_recursive_validation_info.validation_errors)
        
        # 多余的元素
        if len(data) > len(target_type):
            for i in range(len(target_type), len(data)):
                recursive_validation_info.validation_errors.append(ValidationError(
                    path=f"{path}[{i}]",
                    message=f"位置 {path}[{i}] 是多余的元素，类型规范只要求 {len(target_type)} 个元素",
                    expected_type="不应存在",
                    actual_value=data[i],
                    actual_type=type(data[i]),
                    description=description
                ))
    
    return recursive_validation_info

def  _validate_simple_list_type(recursive_validate_args: RecursiveValidateArgs):
    data = recursive_validate_args.data
    target_type = recursive_validate_args.target_type
    path = recursive_validate_args.path
    description = recursive_validate_args.description
    recursive_validation_info = RecursiveValidationInfo()
    if not isinstance(data, list):
            recursive_validation_info.validation_errors.append(ValidationError(
                path=path,
                message=f"位置 {path} 应为列表类型，但实际为{type(data).__name__}类型，值为 {repr(data)}",
                expected_type=list,
                actual_value=data,
                actual_type=type(data),
                description=description
            ))
    return recursive_validation_info

def _validate_list_with_args_type(recursive_validate_args: RecursiveValidateArgs):
    data = recursive_validate_args.data
    target_type = recursive_validate_args.target_type
    path = recursive_validate_args.path
    description = recursive_validate_args.description
    recursive_validation_info = RecursiveValidationInfo()
    if not isinstance(data, list):
        recursive_validation_info.validation_errors.append(ValidationError(
            path=path,
            message=f"位置 {path} 应为列表类型，但实际为{type(data).__name__}类型，值为 {repr(data)}",
            expected_type=target_type,
            actual_value=data,
            actual_type=type(data),
            description=description
        ))
        return recursive_validation_info
    
    element_type = get_args(target_type)
    if len(element_type) == 0:
        return recursive_validation_info
    
    for i, element in enumerate(data):
        element_path = f"{path}[{i}]"
        element_vaidation_info = _validate_recursive(element, element_type[0], element_path)
        recursive_validation_info.validation_errors.extend(element_vaidation_info.validation_errors)
    
    return recursive_validation_info

def _validate_fixed_length_dict(recursive_validate_args: RecursiveValidateArgs):
    data = recursive_validate_args.data
    target_type = recursive_validate_args.target_type
    path = recursive_validate_args.path
    description = recursive_validate_args.description
    target_type_core = {k: v for k, v in target_type.items() if k != ... and v != ...}
    type_pairs = [(k, v) for k, v in target_type.items() if k != ... and v != ...]
    recursive_validation_info = RecursiveValidationInfo()
    
    if len(data) != len(target_type):
        recursive_validation_info.validation_errors.append(ValidationError(
            path=path,
            message=f"位置 {path} 的字典键值对数量应为 {len(target_type)}，但实际为 {len(data)}",
            expected_type=f"包含{len(target_type)}个键值对的字典",
            actual_value=data,
            actual_type=type(data),
            description=description
        ))
    else:
        # breakpoint()
        # 匈牙利算法匹配（简化处理）
        data_items = list(data.items())
        type_items = list(target_type.items())
        
        data_idxs = list(range(len(data_items)))
        type_idxs = list(range(len(type_items)))

        def match_function(data_idx, type_idx):
            key_validate_recursive_info = _validate_recursive(data_items[data_idx][0], type_items[type_idx][0], "")
            value_validate_recursive_info = _validate_recursive(data_items[type_idx][1], type_items[type_idx][1], "")
            return not key_validate_recursive_info.validation_errors and not value_validate_recursive_info.validation_errors
        
        match_result = hungarian_algorithm(data_idxs, type_idxs, match_function)
        
        if match_result is None:
            # 找出无法匹配的项
            for data_key, data_value in data_items:
                key_path = f"{path}[{repr(data_key)}]"
                matched_any = False
                
                for type_key, type_value in type_items:
                    key_validate_recursive_info = _validate_recursive(data_key, type_key, "")
                    value_validate_recursive_info = _validate_recursive(data_value, type_value, "")
                    if not key_validate_recursive_info.validation_errors and not value_validate_recursive_info.validation_errors:
                        matched_any = True
                        break
                
                if not matched_any:
                    recursive_validation_info.validation_errors.append(ValidationError(
                        path=key_path,
                        message=f"{key_path} 无法匹配任何键值对类型要求",
                        expected_type=target_type,
                        actual_value=data,
                        actual_type=f"{type(data)}",
                        description=description
                    ))
    return recursive_validation_info

def _validate_unfixed_length_dict(recursive_validate_args: RecursiveValidateArgs):
    data = recursive_validate_args.data
    target_type = recursive_validate_args.target_type
    path = recursive_validate_args.path
    description = recursive_validate_args.description
    target_type_core = {k: v for k, v in target_type.items() if k != ... and v != ...}
    type_pairs = [(k, v) for k, v in target_type.items() if k != ... and v != ...]
    recursive_validation_info = RecursiveValidationInfo()
    
    for data_key, data_value in data.items():
        key_path = f"{path}[{repr(data_key)}]"
        matched = False
        
        for key_type, value_type in type_pairs:
            key_validate_recursive_info = _validate_recursive(data_key, key_type, f"{key_path}的键")
            value_validate_recursive_info = _validate_recursive(data_value, value_type, key_path)
            
            if not key_validate_recursive_info.validation_errors and not value_validate_recursive_info.validation_errors:
                matched = True
                break
        
        if not matched:
            recursive_validation_info.validation_errors.append(ValidationError(
                path=key_path,
                message=f"位置 {key_path} 的键值对不符合字典的类型要求",
                expected_type=target_type_core,
                actual_value=data,
                actual_type=f"{type(data)}",
                description=description
            ))
    return recursive_validation_info

def _validate_detailed_dict_type(recursive_validate_args: RecursiveValidateArgs):
    data = recursive_validate_args.data
    target_type = recursive_validate_args.target_type
    path = recursive_validate_args.path
    description = recursive_validate_args.description
    recursive_validation_info = RecursiveValidationInfo()
    if not isinstance(data, dict):
        recursive_validation_info.validation_errors.append(ValidationError(
            path=path,
            message=f"位置 {path} 应为字典类型，但实际为{type(data).__name__}类型，值为 {repr(data)}",
            expected_type=dict,
            actual_value=data,
            actual_type=type(data),
            description=description
        ))
        return recursive_validation_info
    
    if len(target_type) == 0:
        return recursive_validation_info
    
    elif ... in target_type and target_type[...] == ...:  # 变长字典
        return _validate_unfixed_length_dict(recursive_validate_args)
    else:  # 定长字典
        return _validate_fixed_length_dict(recursive_validate_args)

def _validate_simple_dict_type(recursive_validate_args: RecursiveValidateArgs):
    data = recursive_validate_args.data
    target_type = recursive_validate_args.target_type
    path = recursive_validate_args.path
    description = recursive_validate_args.description
    recursive_validation_info = RecursiveValidationInfo()
    if not isinstance(data, dict):
        recursive_validation_info.validation_errors.append(ValidationError(
            path=path,
            message=f"位置 {path} 应为字典类型，但实际为{type(data).__name__}类型，值为 {repr(data)}",
            expected_type=dict,
            actual_value=data,
            actual_type=type(data),
            description=description
        ))
    return recursive_validation_info

def _validate_dict_with_args_type(recursive_validate_args: RecursiveValidateArgs):
    data = recursive_validate_args.data
    target_type = recursive_validate_args.target_type
    path = recursive_validate_args.path
    description = recursive_validate_args.description
    recursive_validation_info = RecursiveValidationInfo()
    if not isinstance(data, dict):
        recursive_validation_info.validation_errors.append(ValidationError(
            path=path,
            message=f"位置 {path} 应为字典类型，但实际为{type(data).__name__}类型，值为 {repr(data)}",
            expected_type=target_type,
            actual_value=data,
            actual_type=type(data),
            description=description
        ))
        return recursive_validation_info
    
    key_value_types = get_args(target_type)
    if not key_value_types:
        return recursive_validation_info
    
    key_type, value_type = key_value_types
    
    for data_key, data_value in data.items():
        key_path = f"{path}[{repr(data_key)}]"
        
        key_validate_recursive_info = _validate_recursive(data_key, key_type, f"{key_path}的键")
        value_validate_recursive_info = _validate_recursive(data_value, value_type, key_path)
        
        recursive_validation_info.validation_errors.extend(key_validate_recursive_info.validation_errors)
        recursive_validation_info.validation_errors.extend(value_validate_recursive_info.validation_errors)
    
    return recursive_validation_info

def _validate_union_type(recursive_validate_args: RecursiveValidateArgs):
    data = recursive_validate_args.data
    target_type = recursive_validate_args.target_type
    path = recursive_validate_args.path
    description = recursive_validate_args.description
    recursive_validation_info = RecursiveValidationInfo()
    union_types = get_args(target_type)
    
    for single_type in union_types:
        type_validate_recursive_info = _validate_recursive(data, single_type, path)
        if not type_validate_recursive_info.validation_errors:  # 匹配成功
            return recursive_validation_info
    
    # 所有类型都不匹配
    recursive_validation_info.validation_errors.append(ValidationError(
        path=path,
        message=f"位置 {path} 不符合联合类型的任何选项",
        expected_type=target_type,
        actual_value=data,
        actual_type=type(data),
        description=description
    ))
    return recursive_validation_info

def _validate_unmatched_type(recursive_validate_args: RecursiveValidateArgs):
    data = recursive_validate_args.data
    type_spec = recursive_validate_args.type_spec
    target_type = recursive_validate_args.target_type
    path = recursive_validate_args.path
    description = recursive_validate_args.description
    recursive_validation_info = RecursiveValidationInfo()
    recursive_validation_info.validation_errors.append(ValidationError(
            path=path,
            message=f"位置 {path} 遇到了不支持的类型规范: {repr(type_spec)}",
            expected_type=type_spec,
            actual_value=data,
            actual_type=type(data),
            description=description
        ))
    return recursive_validation_info


def _validate_recursive(original_data, type_spec, path="data") -> RecursiveValidationInfo:
    """
    递归验证函数，返回错误列表。
    不允许修改 original_data 的值。但是有时候需要修正 data 的值，因此应当将它存储在 converted data 中。
    """
    data = deepcopy(original_data)
    if isinstance(type_spec, tuple):
        description, target_type = type_spec
    else:
        description, target_type = None, type_spec
    
    recursive_validate_args = RecursiveValidateArgs(
        data=data,
        type_spec=type_spec,
        target_type=target_type,
        path=path,
        description=description
    )
    
    # 基础类型
    if isinstance(target_type, basic_types) or target_type in basic_values:  # value type
        return _validate_value_types(recursive_validate_args)
    elif target_type in basic_types:  # other basic type
        return _validate_basic_types(recursive_validate_args)
    
    # 列表类型
    elif isinstance(target_type, list):
        return _validate_detailed_list_type(recursive_validate_args)
    elif target_type == list:
        return _validate_simple_list_type(recursive_validate_args)
    elif get_origin(target_type) == list:  # list[SomeType]
        return _validate_list_with_args_type(recursive_validate_args)
    
    # 字典类型处理（类似的结构，但创建ValidationError对象）
    elif isinstance(target_type, dict):
        return _validate_detailed_dict_type(recursive_validate_args)
    elif target_type == dict:
        return _validate_simple_dict_type(recursive_validate_args)
    elif get_origin(target_type) == dict:  # dict[KeyType, ValueType]
        return _validate_dict_with_args_type(recursive_validate_args)
    
    # Union类型
    elif get_origin(target_type) == UnionType:
        return _validate_union_type(recursive_validate_args)
    else:
        return _validate_unmatched_type(recursive_validate_args)

def validate_type_with_result(data, type_spec, path="data") -> ValidationResult:
    """
    递归验证类型，返回 ValidationResult 对象
    
    Args:
        data: 要验证的数据
        type_spec: 类型规范
        path: 当前数据路径，用于错误定位
    
    Returns:
        ValidationResult: 包含详细验证结果的对象
    """
    
    # 执行验证
    recursive_validation_info = _validate_recursive(data, type_spec, path)
    converted_data = recursive_validation_info.get_converted_data(original_data=data)
    
    validation_errors = recursive_validation_info.validation_errors
    return ValidationResult(
        is_valid=recursive_validation_info.is_valid,
        errors=validation_errors,
        data=data,
        type_spec=type_spec,
        is_converted=recursive_validation_info.need_convert,
        converted_data=converted_data,
        convert_results=recursive_validation_info.convert_results
    )


def validate_test():
    # 基础类型
    assert validate_type("test", ("value", "test")) == True 
    assert validate_type("Alice", ("name", str)) == True 
    assert validate_type(25, ("age", int)) == True 
    assert validate_type("25", ("active", bool)) == False 

    # 列表类型
    assert validate_type([["math", 95], ["english", 87]], [[("subject", str), ("score", int)], ...]) == True 
    assert validate_type([["math", 95], ["english", 87]], ("scores", list[list[str | int]])) == True

    # 字典类型 Dict[TupleSpec, TupleSpec]
    assert validate_type({"math": 95, "english": 87}, ("scores", {"math": int, "english": int})) == True 
    assert validate_type({"math": 95, "english": 87}, ("scores", {("subject", str): ("score", int), ...:...}))  == True # ... 表 示 前面声明的类型若干

    # 复杂嵌套示例
    assert validate_type([{"name": "Alice", "phone" : "123-456-7890"},{"name": "Bob", "phone": "098-765-4321"}], ("friends", [("friend", {"name": str, "phone": str}), ...])) == True

    assert validate_type([
        "Alice",
        25.1,
        ["123 Main St", "New York"],
        [
            {"name": "Bob", "phone": "123-456-7890"},
            {"name": "Charlie", "phone": "123-456-7890"}
        ]
    ], ("person", [("name", str), ("age", float), ("address", [("street", str), ("city", str)]), ("friends", [("friend", {"name": str, "phone": str}), ...])]))  == True

def explain_test():

    print(explain_with_example(("value", "test")))
    print(explain_with_example(("name", str)))
    print(explain_with_example(("age", int)))
    print(explain_with_example(("active", bool)))

    # 列表类型
    print(explain_with_example([[("subject", str), ("score", int)], ...])) 
    print(explain_with_example(("scores", list[list[str | int]])))

    # 字典类型 Dict[TupleSpec, TupleSpec]
    print(explain_with_example(("scores", {"math": int, "english": int}))) 
    print(explain_with_example(("scores", {("subject", str): ("score", int), ...:...}))) # ... 表示前面声明的类型若干

    # 复杂嵌套示例
    print(explain_with_example(("friends", [("friend", {"name": str, "phone": str}), ...])))

    print(explain_with_example(("person", [("name", str), ("age", float), ("address", [("street", str), ("city", str)]), ("friends", [("friend", {"name": str, "phone": str}), ...])])))


def validate_with_result_test():
    
    def assert_validation(result: ValidationResult, expected_valid: bool, test_name="", expected_error_count=None, expected_error_paths=None):
        """
        断言验证结果
        
        Args:
            result: 验证结果
            expected_valid: 期望的验证结果（True/False）
            test_name: 测试名称
            expected_error_count: 期望的错误数量（可选）
            expected_error_paths: 期望的错误路径列表（可选）
        """
        print(f"=== {test_name} ===")
        explain_type(result.type_spec)
        generate_json_example(result.type_spec)
        # 基本验证结果断言
        try:
            assert result.success == expected_valid, f"期望验证结果为 {expected_valid}，但实际为 {result.success}"
            
            if expected_valid:
                assert result.error_count == 0, f"期望无错误，但发现 {result.error_count} 个错误"
                print("✓ 验证通过 - 符合预期")
            else:
                assert result.error_count > 0, f"期望有错误，但实际无错误"
                
                
                print(f"✓ 验证失败 - 符合预期（{result.error_count} 个错误）")
                
                # 显示错误详情（简化版）
                for i, error in enumerate(result.errors, 1):
                    print(f"  {i}. {error.path}: {error.message}")
            
            print("✅ 断言通过\n")
            
        except AssertionError as e:
            print(f"❌ 断言失败: {e}")
            print(f"实际验证结果: {result.success}")
            print(f"实际错误数量: {result.error_count}")
            if result.errors:
                print("实际错误:")
                for error in result.errors:
                    print(f"  - {error.path}: {error.message}")
            print()
            raise

    print("开始类型验证测试...\n")

    # 1. 基础类型测试
    print("【基础类型测试】")
    
    # 正确的情况
    assert_validation(
        validate_type_with_result("test", ("value", "test")), 
        expected_valid=True, 
        test_name="常量值匹配"
    )
    
    assert_validation(
        validate_type_with_result("Alice", ("name", str)), 
        expected_valid=True, 
        test_name="字符串类型"
    )
    
    assert_validation(
        validate_type_with_result(25, ("age", int)), 
        expected_valid=True, 
        test_name="整数类型"
    )
    
    assert_validation(
        validate_type_with_result(True, ("active", bool)), 
        expected_valid=True, 
        test_name="布尔类型"
    )
    
    # 错误的情况
    assert_validation(
        validate_type_with_result("wrong", ("value", "test")), 
        expected_valid=False, 
        expected_error_count=1,
        expected_error_paths=["data"],
        test_name="常量值不匹配"
    )
    
    assert_validation(
        validate_type_with_result(123, ("name", str)), 
        expected_valid=False, 
        expected_error_count=1,
        expected_error_paths=["data"],
        test_name="类型不匹配-应为字符串"
    )
    
    assert_validation(
        validate_type_with_result("25", ("age", int)), 
        expected_valid=True, 
        expected_error_count=1,
        expected_error_paths=["data"],
        test_name="类型匹配-转换为整数"
    )
    
    assert_validation(
        validate_type_with_result("true", ("active", bool)), 
        expected_valid=True, 
        expected_error_count=1,
        expected_error_paths=["data"],
        test_name="类型匹配-转换为为布尔"
    )

    # 2. 列表类型测试
    print("【列表类型测试】")
    
    # 变长列表 - 正确
    assert_validation(
        validate_type_with_result([["math", 95], ["english", 87]], 
                                [[("subject", str), ("score", int)], ...]), 
        expected_valid=True,
        test_name="变长列表-正确"
    )
    
    # 变长列表 - 错误
    assert_validation(
        validate_type_with_result([["math", 95], ["english", "87"]], 
                                [[("subject", str), ("score", int)], ...]), 
        expected_valid=False,
        expected_error_count=1,
        expected_error_paths=["data[1][1]"],
        test_name="变长列表-元素类型错误"
    )
    
    assert_validation(
        validate_type_with_result([["math", 95], [123, 87]], 
                                [[("subject", str), ("score", int)], ...]), 
        expected_valid=False,
        expected_error_count=1,
        expected_error_paths=["data[1][0]"],
        test_name="变长列表-多个元素错误"
    )
    
    # 泛型列表 - 正确
    assert_validation(
        validate_type_with_result([["math", 95], ["english", 87]], 
                                ("scores", list[list[str | int]])), 
        expected_valid=True,
        test_name="泛型列表-正确"
    )
    
    # 泛型列表 - 错误
    assert_validation(
        validate_type_with_result([["math", 95], ["english", 87.5, True]], 
                                ("scores", list[list[str | int]])), 
        expected_valid=False,
        expected_error_count=1,  # True 不匹配 str | int
        expected_error_paths=["data[1][2]"],
        test_name="泛型列表-包含不支持类型"
    )
    
    # 定长列表 - 错误
    assert_validation(
        validate_type_with_result([1, 2, 3, 4], 
                                [("first", int), ("second", str), ("third", bool)]), 
        expected_valid=False,
        expected_error_count=3,  # 长度不匹配 + 第2个元素类型错误 + 第3个元素类型错误 + 多余元素
        expected_error_paths=["data", "data[1]", "data[2]", "data[3]"],
        test_name="定长列表-长度不匹配"
    )

    # 3. 字典类型测试
    print("【字典类型测试】")
    
    # 定长字典 - 正确
    assert_validation(
        validate_type_with_result({"math": 95, "english": 87}, 
                                ("scores", {"math": int, "english": int})), 
        expected_valid=True,
        test_name="定长字典-正确"
    )
    
    # 定长字典 - 错误
    assert_validation(
        validate_type_with_result({"math": "95", "english": 87}, 
                                ("scores", {"math": int, "english": int})), 
        expected_valid=False,
        expected_error_count=1,  # math的值类型错误
        test_name="定长字典-值类型错误"
    )
    
    assert_validation(
        validate_type_with_result({"math": 95, "science": 87}, 
                                ("scores", {"math": int, "english": int})), 
        expected_valid=False,
        expected_error_count=1,  # science键无法匹配
        test_name="定长字典-键不匹配"
    )
    
    assert_validation(
        validate_type_with_result({"math": 95}, 
                                ("scores", {"math": int, "english": int})), 
        expected_valid=False,
        expected_error_count=1,  # 字典长度不匹配
        expected_error_paths=["data"],
        test_name="定长字典-缺少键"
    )
    
    # 变长字典 - 正确
    assert_validation(
        validate_type_with_result({"math": 95, "english": 87, "science": 92}, 
                                ("scores", {("subject", str): ("score", int), ...: ...})), 
        expected_valid=True,
        test_name="变长字典-正确"
    )
    
    # 变长字典 - 错误
    assert_validation(
        validate_type_with_result({"math": 95, 123: 87}, 
                                ("scores", {("subject", str): ("score", int), ...: ...})), 
        expected_valid=False,
        expected_error_count=1,  # 123键类型错误
        expected_error_paths=["data[123]"],
        test_name="变长字典-键类型错误"
    )
    
    # 泛型字典 - 错误
    assert_validation(
        validate_type_with_result({"name": "Alice", "age": "25"}, 
                                dict[str, int]), 
        expected_valid=False,
        expected_error_count=1,  # age的值类型错误
        expected_error_paths=["data['age']"],
        test_name="泛型字典-值类型错误"
    )

    assert_validation(
        validate_type_with_result({"name": "Alice", "age": 25, "phone": None}, 
                                dict[str, int | str | None]), 
        expected_valid=True,
        test_name="泛型字典"
    )
    
    # 4. 复杂嵌套测试
    print("【复杂嵌套测试】")
    
    # 正确的复杂结构
    assert_validation(
        validate_type_with_result([{"name": "Alice", "phone": "123-456-7890"}, 
                                 {"name": "Bob", "phone": "098-765-4321"}], 
                                ("friends", [("friend", {"name": str, "phone": str}), ...])), 
        expected_valid=True,
        test_name="嵌套结构-正确"
    )

    assert_validation(
        validate_type_with_result([{"name": "Alice", "phone": "123-456-7890"}, 
                                 {"name": "Bob", "phone": None}], 
                                ("friends", [("friend", {"name": str, "phone": str | None}), ...])), 
        expected_valid=True,
        test_name="嵌套结构-正确"
    )
    
    # 错误的复杂结构 - 多个错误
    assert_validation(
        validate_type_with_result([{"name": "Alice", "phone": 1234567890}, 
                                 {"name": 123, "phone": "098-765-4321"}], 
                                ("friends", [("friend", {"name": str, "phone": str}), ...])), 
        expected_valid=False,
        expected_error_count=2,  # 第一个phone类型错误，第二个name类型错误
        test_name="嵌套结构-多个错误"
    )
    
    # 超级复杂的嵌套结构 - 包含多种错误
    complex_data = [
        "Alice",           # 正确
        25.1,             # 正确
        ["123 Main St", 123],  # 错误：city应为字符串
        [
            {"name": "Bob", "phone": "123-456-7890"},      # 正确
            {"name": "Charlie", "phone": "123-456-7890"},  # 正确
            {"name": 2123, "phone": "123-456-7890"}        # 错误：name应为字符串
        ]
    ]
    
    complex_spec = ("person", [
        ("name", str), 
        ("age", float), 
        ("address", [("street", str), ("city", str)]), 
        ("friends", [("friend", {"name": str, "phone": str}), ...])
    ])
    
    assert_validation(
        validate_type_with_result(complex_data, complex_spec), 
        expected_valid=False,
        expected_error_count=2,  # address[1]类型错误 + friends[2].name类型错误
        expected_error_paths=["data[2][1]"],  # 至少包含这个路径
        test_name="超复杂嵌套-多层错误"
    )

    # 5. Union类型测试
    print("【Union类型测试】")
    
    # Union类型 - 正确
    assert_validation(
        validate_type_with_result("hello", str | int | None), 
        expected_valid=True,
        test_name="Union类型-匹配第一个"
    )
    
    assert_validation(
        validate_type_with_result(42, str | int | None), 
        expected_valid=True,
        test_name="Union类型-匹配第二个"
    )

    assert_validation(
        validate_type_with_result(None, str | int | None), 
        expected_valid=True,
        test_name="Union类型-匹配第三个"
    )
    # Union类型 - 错误
    assert_validation(
        validate_type_with_result(3.14, str | int), 
        expected_valid=False,
        expected_error_count=1,
        expected_error_paths=["data"],
        test_name="Union类型-都不匹配"
    )
    
    # 6. 边界情况测试
    print("【边界情况测试】")
    
    # 空列表和空字典
    assert_validation(
        validate_type_with_result([], list), 
        expected_valid=True,
        test_name="空列表"
    )
    
    assert_validation(
        validate_type_with_result({}, dict), 
        expected_valid=True,
        test_name="空字典"
    )
    
    assert_validation(
        validate_type_with_result([], [int, ...]), 
        expected_valid=True,
        test_name="空列表vs变长列表"
    )
    
    # None值
    assert_validation(
        validate_type_with_result(None, str), 
        expected_valid=False,
        expected_error_count=1,
        expected_error_paths=["data"],
        test_name="None值"
    )
    
    # 类型不匹配的根本错误
    assert_validation(
        validate_type_with_result("not a list", [int, ...]), 
        expected_valid=False,
        expected_error_count=1,
        expected_error_paths=["data"],
        test_name="根本类型错误"
    )
    
    assert_validation(
        validate_type_with_result("not a dict", {"key": str}), 
        expected_valid=False,
        expected_error_count=1,
        expected_error_paths=["data"],
        test_name="根本类型错误-字典"
    )

    # 7. 额外的边界测试
    print("【额外边界测试】")
    
    # 测试空的类型规范
    assert_validation(
        validate_type_with_result([], []), 
        expected_valid=True,
        test_name="空类型规范-列表"
    )
    
    assert_validation(
        validate_type_with_result({}, {}), 
        expected_valid=True,
        test_name="空类型规范-字典"
    )
    
    # 测试嵌套的空结构
    assert_validation(
        validate_type_with_result([[], {}], [list, dict]), 
        expected_valid=True,
        test_name="嵌套空结构"
    )
    
    # 测试多层嵌套错误
    assert_validation(
        validate_type_with_result([[[["wrong"]]]], [[[int]]]), 
        expected_valid=False,
        expected_error_count=1,
        expected_error_paths=["data[0][0][0][0]"],
        test_name="深层嵌套错误"
    )
    
    assert_validation(
        validate_type_with_result({'format_name': 'Pipe_Delimited_Order', 'description': 'A pipe-delimited text format for order transactions, commonly used to export order data with basic details and metadata.', 'key_fields': ['DATE', 'ORDER_ID', 'USER_ID', 'PRODUCT_LIST', 'TOTAL', 'STATUS', 'CREATED_DATE'], 'example_pattern': 'ORD|20250115|O456789|user_456|产品A*2+产品B*1|total:¥299.50|status:pending|created:2025-01-15T10:30:00+08:00'}, {'format_name': str, 'description': str, 'key_fields': [str, ...], 'example_pattern': str}), 
        expected_valid=True,
        test_name="完美匹配测试"
    )

    assert_validation(
        validate_type_with_result(
            "123",
            int,
        ),
        expected_valid=True,
        test_name="str -> int 转换测试"
    )

    assert_validation(
        validate_type_with_result(
            "123.1",
            float,
        ),
        expected_valid=True,
        test_name="str -> float 转换测试"
    )

    assert_validation(
        validate_type_with_result(
            "123.1",
            int,
        ),
        expected_valid=False,
        test_name="str -> int 转换测试（错误测试）"
    )

    assert_validation(
        validate_type_with_result(
            "awefx",
            int,
        ),
        expected_valid=False,
        test_name="str -> int 转换测试（错误测试）"
    )

    assert_validation(
        validate_type_with_result(
            "False",
            bool,
        ),
        expected_valid=True,
        test_name="str -> bool 转换测试"
    )

    assert_validation(
        validate_type_with_result(
            "awefx",
            bool,
        ),
        expected_valid=False,
        test_name="str -> bool 转换测试（错误测试）"
    )

    assert_validation(
        validate_type_with_result(
            1.3,
            str,
        ),
        expected_valid=False,
        test_name="float -> str 转换测试（错误测试）"
    )

    assert_validation(
        validate_type_with_result(
            1,
            str,
        ),
        expected_valid=False,
        test_name="int -> str 转换测试（错误测试）"
    )

    assert_validation(
        validate_type_with_result(
            False,
            str,
        ),
        expected_valid=False,
        test_name="bool -> str 转换测试（错误测试）"
    )

    print("✅ 所有测试通过！验证函数工作正常，既不遗漏错误也不误报。")

if __name__ == "__main__":
    # validate_test()
    # explain_test()
    # validate_with_result_test()
    result = validate_type_with_result("true", bool)
    converted_data = result.converted_data
    print(result.is_valid)
    print(result.is_converted)
    print(converted_data, type(converted_data))
    # res = explain_with_example({col: (col, str | int | float | None) for col in ['transaction_id','order_id', 'reference_id', 'user_id', 'customer_name','status', 'transaction_type', 'method', 'amount', 'currency']})
    # print(res)