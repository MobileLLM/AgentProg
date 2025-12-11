import ast
import re
import sys
import functools
from agentprog.all_utils.general_utils import init_get_openai_response

class ReturnReplacer(ast.NodeTransformer):
    def __init__(self, ret_val_name='__return_value__'):
        self.ret_val_name = ret_val_name
        self.function_stack = []
        self.replacements = []  # 记录替换信息
        self.source_lines = []
    
    def visit_FunctionDef(self, node):
        self.function_stack.append(node.name)
        self.generic_visit(node)
        self.function_stack.pop()
        return node
    
    def visit_AsyncFunctionDef(self, node):
        self.function_stack.append(node.name)
        self.generic_visit(node)
        self.function_stack.pop()
        return node
    
    def visit_Return(self, node):
        if not self.function_stack:
            # 在函数外的return，需要替换
            if node.value is None:
                # return 没有值的情况
                eval_str = "None"
            else:
                # 将返回值的AST转换回代码字符串
                eval_str = ast.unparse(node.value)
            
            # 记录替换信息
            self.replacements.append({
                'line': node.lineno,
                'col': node.col_offset,
                'original': f"return {eval_str}" if node.value else "return",
                'replacement': f"globals()['{self.ret_val_name}'] = {eval_str}"
            })
            
            # 创建新的赋值节点
            # globals()['__return_value__'] = value
            new_node = ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Call(
                            func=ast.Name(id='globals', ctx=ast.Load()),
                            args=[],
                            keywords=[]
                        ),
                        attr='__setitem__',
                        ctx=ast.Load()
                    ),
                    args=[
                        ast.Constant(value=self.ret_val_name),
                        node.value if node.value else ast.Constant(value=None)
                    ],
                    keywords=[]
                )
            )
            
            # 复制位置信息
            ast.copy_location(new_node, node)
            return new_node
        
        return node

class ReturnProcessor:
    def __init__(self, ret_val_name='__return_value__'):
        self.ret_val_name = ret_val_name
    
    def check_and_replace(self, code_string):
        """
        检查并替换代码中的非法return语句
        
        Args:
            code_string (str): 原始代码字符串
            
        Returns:
            dict: 包含处理结果的字典
        """
        try:
            # 解析代码为AST
            tree = ast.parse(code_string)
            
            # 创建替换器
            replacer = ReturnReplacer(self.ret_val_name)
            replacer.source_lines = code_string.split('\n')
            
            # 执行替换
            new_tree = replacer.visit(tree)
            
            # 生成新的代码
            new_code = ast.unparse(new_tree)
            
            return {
                'success': True,
                'original_code': code_string,
                'modified_code': new_code,
                'replacements': replacer.replacements,
                'has_replacements': len(replacer.replacements) > 0
            }
            
        except SyntaxError as e:
            return {
                'success': False,
                'error': f"Syntax error: {e.msg}",
                'line': e.lineno,
                'offset': e.offset
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def process_with_string_replacement(self, code_string):
        """
        使用字符串替换的方式处理return语句（备用方法）
        """
        lines = code_string.split('\n')
        modified_lines = []
        replacements = []
        
        # 简单的函数检测（这个方法不够精确，主要用于演示）
        in_function = False
        indent_level = 0
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # 检测函数定义
            if stripped.startswith('def ') or stripped.startswith('async def '):
                in_function = True
                indent_level = len(line) - len(line.lstrip())
                modified_lines.append(line)
                continue
            
            # 检测函数结束（简单的缩进检测）
            if in_function and line.strip() and len(line) - len(line.lstrip()) <= indent_level:
                if not (stripped.startswith('def ') or stripped.startswith('async def ')):
                    in_function = False
            
            # 处理return语句
            if stripped.startswith('return') and not in_function:
                # 提取return后的表达式
                return_match = re.match(r'(\s*)return\s*(.*)', line)
                if return_match:
                    indent, expr = return_match.groups()
                    if not expr.strip():
                        expr = 'None'
                    
                    new_line = f"{indent}globals()['{self.ret_val_name}'] = {expr}"
                    modified_lines.append(new_line)
                    
                    replacements.append({
                        'line': i,
                        'original': line.strip(),
                        'replacement': new_line.strip()
                    })
                else:
                    modified_lines.append(line)
            else:
                modified_lines.append(line)
        
        return {
            'success': True,
            'original_code': code_string,
            'modified_code': '\n'.join(modified_lines),
            'replacements': replacements,
            'has_replacements': len(replacements) > 0
        }

def test_return_replacement():
    """测试return替换功能"""
    processor = ReturnProcessor('__return_value__')
    
    test_cases = [
        # 测试用例1: 简单的函数外return
        """
x = 10
return x
print("after return")
        """,
        
        # 测试用例2: 混合情况
        """
def valid_func():
    return "valid"

x = 20
return x + 5

def another_func():
    return "also valid"

return "final return"
        """,
        
        # 测试用例3: 复杂表达式
        """
data = [1, 2, 3]
return sum(data) * 2
        """,
        
        # 测试用例4: 无值return
        """
print("before")
return
print("after")
        """,
        
        # 测试用例5: 嵌套函数
        """
def outer():
    def inner():
        return "inner"
    return inner()

return "outer return"
        """
    ]
    
    for i, code in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"测试用例 {i}:")
        print("原始代码:")
        print(code.strip())
        
        result = processor.check_and_replace(code)
        
        if result['success']:
            print(f"\n处理结果: {'有替换' if result['has_replacements'] else '无需替换'}")
            
            if result['has_replacements']:
                print("\n替换详情:")
                for repl in result['replacements']:
                    print(f"  第{repl['line']}行: {repl['original']} -> {repl['replacement']}")
                
                print("\n修改后的代码:")
                print(result['modified_code'])
                
                # 测试执行修改后的代码
                print("\n执行结果:")
                try:
                    exec_globals = {}
                    exec(result['modified_code'], exec_globals)
                    if '__return_value__' in exec_globals:
                        print(f"返回值: {exec_globals['__return_value__']}")
                    else:
                        print("无返回值")
                except Exception as e:
                    print(f"执行错误: {e}")
        else:
            print(f"处理失败: {result['error']}")

# 便捷函数
def replace_returns(code_string, ret_val_name='__return_value__'):
    """
    便捷函数：替换代码中的非法return语句
    
    Args:
        code_string (str): 原始代码
        ret_val_name (str): 返回值变量名
    
    Returns:
        str: 修改后的代码
    """
    processor = ReturnProcessor(ret_val_name)
    result = processor.check_and_replace(code_string)
    
    if result['success']:
        return result['modified_code']
    else:
        raise ValueError(f"代码处理失败: {result['error']}")

import ast
import re

class BreakChecker(ast.NodeVisitor):
    def __init__(self, break_flag_name="should_break"):
        self.break_flag_name = break_flag_name
        self.loop_stack = []  # 跟踪当前的循环嵌套
        self.replacements = []  # 存储需要替换的break位置
        
    def visit_For(self, node):
        self.loop_stack.append(node)
        self.generic_visit(node)
        self.loop_stack.pop()
        
    def visit_While(self, node):
        self.loop_stack.append(node)
        self.generic_visit(node)
        self.loop_stack.pop()
        
    def visit_Break(self, node):
        if not self.loop_stack:  # break不在任何循环内
            self.replacements.append({
                'line': node.lineno,
                'col': node.col_offset,
                'replacement': f"{self.break_flag_name} = True"
            })

def check_and_replace_breaks(code_string, break_flag_name="__break_flag_name__"):
    """
    检查代码中的break语句，如果不在循环内则替换为标志变量赋值
    
    Args:
        code_string: 要检查的Python代码字符串
        break_flag_name: 替换break时使用的标志变量名
    
    Returns:
        tuple: (修改后的代码, 是否有修改)
    """
    try:
        # 解析代码为AST
        tree = ast.parse(code_string)
        
        # 检查break语句
        checker = BreakChecker(break_flag_name)
        checker.visit(tree)
        
        if not checker.replacements:
            return code_string, False
            
        # 按行号倒序排列，避免替换时行号偏移
        checker.replacements.sort(key=lambda x: x['line'], reverse=True)
        
        # 分割代码为行
        lines = code_string.split('\n')
        
        # 执行替换
        for replacement in checker.replacements:
            line_idx = replacement['line'] - 1  # AST行号从1开始
            if line_idx < len(lines):
                line = lines[line_idx]
                # 找到break关键字并替换
                # 使用正则表达式确保只替换独立的break关键字
                pattern = r'\bbreak\b'
                lines[line_idx] = re.sub(pattern, replacement['replacement'], line)
        
        return '\n'.join(lines), True
        
    except SyntaxError as e:
        print(f"语法错误: {e}")
        return code_string, False

# 测试函数
def test_break_replacement():
    test_cases = [
        # 测试用例1: break在循环外
        """
def func():
    if condition:
        break
    return True
""",
        
        # 测试用例2: break在循环内（不应该替换）
        """
for i in range(10):
    if i == 5:
        break
    print(i)
""",
        
        # 测试用例3: 混合情况
        """
def complex_func():
    for i in range(10):
        if i == 3:
            break  # 这个不应该替换
    
    if some_condition:
        break  # 这个应该替换
    
    while True:
        if condition:
            break  # 这个不应该替换
        else:
            continue
    
    # 另一个循环外的break
    if error:
        break  # 这个应该替换
""",
        
        # 测试用例4: 嵌套循环
        """
for i in range(5):
    for j in range(5):
        if j == 2:
            break  # 内层循环的break，不应该替换
    if i == 3:
        break  # 外层循环的break，不应该替换

# 循环外的break
if error_occurred:
    break  # 这个应该替换
"""
    ]
    
    for i, test_code in enumerate(test_cases, 1):
        print(f"\n=== 测试用例 {i} ===")
        print("原代码:")
        print(test_code)
        
        modified_code, was_modified = check_and_replace_breaks(test_code, "break_flag")
        
        print(f"\n是否有修改: {was_modified}")
        if was_modified:
            print("修改后的代码:")
            print(modified_code)

def replace_break(code_string, break_flag_name="__break_flag_name__"):
    modified_code, _ = check_and_replace_breaks(code_string, break_flag_name)
    return modified_code

@functools.lru_cache
def convert_for_loop_to_iterator(loop_str) -> tuple[str, str]:
    get_openai_response = init_get_openai_response()
    from agentprog.all_utils.fm import get_default_fm
    llm = get_default_fm(get_response=get_openai_response)
    res = llm.query(f"""
要求：输入一个 for 循环语句，将其改造成迭代器取数。例如：for i, expense_entry in enumerate(ui.iterate_views("Expense entry, showing amount in large text, date and time below, description text, and category tag")):

改写成
1. iterator: i_expense_entry_iterator = iter(enumerate(ui.iterate_views("Expense entry, showing amount in large text, date and time below, description text, and category tag")))
2. get_next: i, expense_entry = next(i_expense_entry_iterator)

注意：
1. 严格遵照命名规则，迭代器的命名方式为 <variable_name>_iterator，若变量有多个，用下划线按照顺序连接，构成其对应的 iterator。例如 i_expense_entry_iterator，就是表示其取出的变量名为 i, expense_entry。

现在，你的输入是：
{loop_str}

""", returns={"iterator": str, "get_next": str})
    return res['iterator'], res['get_next']

def test_converter():
    """测试转换器"""
    test_cases = [
        'for i, expense_entry in enumerate(ui.iterate_views("Expense entry, showing amount in large text, date and time below, description text, and category tag")):',
        'for item in my_list:',
        'for key, value in my_dict.items():',
        'for x, y, z in zip(list1, list2, list3):',
        'for i in range(10):',
        'for line in open("file.txt"):',
        # 包含多个 in 的复杂情况
        'for item in [x for x in data if "in" in x]:',
        'for key in {"key_in_dict": "value", "another": "in"}:',
        'for item in [x for x in data if "in" in x]:',
        'for i, expense_entry in enumerate(ui.iterate_views("Expense entry, showing amount in large text, date and time below, description text, and category tag")):',
    ]
    for test_case in test_cases:
        iterator, get_next = convert_for_loop_to_iterator(test_case)
        print(f"原始语句：")
        print(f"{test_case}")
        print(f"转换结果：")
        print(iterator)
        print(get_next)
        print("-" * 80)

if __name__ == "__main__":
    # 运行测试
    test_break_replacement()