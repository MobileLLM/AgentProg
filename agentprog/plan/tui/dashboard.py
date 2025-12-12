import json
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.syntax import Syntax
from rich.layout import Layout
from rich import box

# --- 1. 将所有常量字符串单独提取出来 ---
GOAL = "GOAL"
WORKFLOW_CONTEXT = "WORKFLOW CONTEXT"
CODE_CONTEXT = "CODE CONTEXT"
DATA_AND_VARIABLES = "DATA AND VARIABLES"
BELIEF_STATE = "BELIEF STATE"
PLAN = "PLAN"
ACTION = "ACTION"

# --- 保留的通用字段常量 ---
STEP_ID = "STEP ID"
OBSERVATION = "OBSERVATION"
THOUGHT = "THOUGHT"


class AgentLogVisualizer:
    def __init__(self):
        self.console = Console()
        # --- 使用常量初始化 styles ---
        self.styles = {
            GOAL: ("bright_green", "🎯"),
            WORKFLOW_CONTEXT: ("cyan", "🌐"),
            CODE_CONTEXT: ("bright_white", "💻"),
            DATA_AND_VARIABLES: ("yellow", "📊"),
            BELIEF_STATE: ("bright_blue", "📦"),
            PLAN: ("bright_magenta", "📋"),
            ACTION: ("bright_white", "⚡"),
            # --- 通用字段 ---
            STEP_ID: ("bright_cyan", "👉"),
            OBSERVATION: ("bright_green", "🔍"),
            THOUGHT: ("bright_yellow", "🧠"),
        }

    def _get_content_renderable(self, key, content, color, folded=False):
        """根据内容类型创建可渲染的 Panel。"""
        icon = self.styles.get(key, ("white", ""))[1]
        title = f"{icon}  {key}"

        if not content:
            content = " "

        # 1. JSON/字典/列表 处理
        if isinstance(content, (dict, list)):
            indent = 1 if folded else 2
            try:
                json_str = json.dumps(content, indent=indent, ensure_ascii=False)
            except TypeError:
                json_str = str(content)
            syntax = Syntax(json_str, "json", theme="light", line_numbers=True, word_wrap=True)
            return Panel(syntax, title=title, border_style=color, box=box.ROUNDED)

        # 2. 文本及代码处理
        else:
            txt_str = str(content)
            # --- 2. WORKFLOW CONTEXT 和 CODE CONTEXT 都用代码风格显示 ---
            # 同时也将其他适合代码风格的字段包含进来
            if key in [WORKFLOW_CONTEXT, CODE_CONTEXT, DATA_AND_VARIABLES, ACTION]:
                syntax = Syntax(txt_str, "python", theme="light", line_numbers=True, word_wrap=True)
                return Panel(syntax, title=title, border_style=color, box=box.ROUNDED)

            # 默认文本处理
            txt = Text(txt_str, no_wrap=False)
            txt.highlight_words(["CRITICAL_ERROR", "incorrect", "must be corrected", "error", "failed"], style="bold red on yellow")
            return Panel(txt, title=title, border_style=color, box=box.ROUNDED)

    def show(self, show_data, folded=True):
        """主显示函数。"""
        # --- 使用常量进行键名映射 ---
        key_mapping = {
            "goal": GOAL,
            "workflow_context_str": WORKFLOW_CONTEXT,
            "code_context": CODE_CONTEXT,
            "data_and_variables": DATA_AND_VARIABLES,
            "belief_state": BELIEF_STATE,
            "plan": PLAN,
            "action": ACTION,
        }
        
        display_data = {
            style_key: show_data.get(data_key) 
            for data_key, style_key in key_mapping.items()
        }

        if folded:
            self.console.clear()
            self._render_folded(display_data)
        else:
            # self.console.clear()
            self._render_unfolded(display_data)

    def _render_unfolded(self, data):
        """流式打印模式：完整打印所有内容。"""
        order = [GOAL, WORKFLOW_CONTEXT, CODE_CONTEXT, DATA_AND_VARIABLES, BELIEF_STATE, PLAN, ACTION]
        self.console.rule("[bold cyan]Start of Step[/bold cyan]", style="cyan")
        for key in order:
            if key in data and data[key]:
                color = self.styles[key][0]
                renderable = self._get_content_renderable(key, data[key], color, folded=False)
                self.console.print(renderable)
        self.console.rule("[bold cyan]End of Step[/bold cyan]", style="cyan")

    def _render_folded(self, data):
        """Dashboard 模式，采用您提供的最新布局。"""
        layout = Layout()
        
        # --- 布局名称也使用常量的小写版本，更规范 ---
        layout.split_column(
            Layout(name="padding", size=1),
            Layout(name=GOAL.lower(), size=3),
            Layout(name=WORKFLOW_CONTEXT.lower().replace(" ", "_"), size=8),
            Layout(name=CODE_CONTEXT.lower().replace(" ", "_"), size=8),
            Layout(name="main_body")
        )

        layout["main_body"].split_column(
            Layout(name="state_info", ratio=3),
            Layout(name="decision_info", ratio=1)
        )
        
        layout["state_info"].split_row(
            Layout(name="data_vars"),
            Layout(name="belief")
        )
        
        layout["decision_info"].split_row(
            Layout(name="plan"),
            Layout(name="action")
        )

        # --- 填充内容时使用常量 ---
        def get_panel(k):
            color = self.styles.get(k, ("white", ""))[0]
            content = data.get(k)
            return self._get_content_renderable(k, content, color, folded=True)

        layout[GOAL.lower()].update(get_panel(GOAL))
        layout[WORKFLOW_CONTEXT.lower().replace(" ", "_")].update(get_panel(WORKFLOW_CONTEXT))
        layout[CODE_CONTEXT.lower().replace(" ", "_")].update(get_panel(CODE_CONTEXT))
        layout["data_vars"].update(get_panel(DATA_AND_VARIABLES))
        layout["belief"].update(get_panel(BELIEF_STATE))
        layout["plan"].update(get_panel(PLAN))
        layout["action"].update(get_panel(ACTION))

        self.console.print(layout)

if __name__ == "__main__":
    viz = AgentLogVisualizer()
    
    sample_show_data = {
        'goal': '', 
        'workflow_context_str': 'In the `Contacts` app, create a new contact with the name "agent prog" and save it.  # <-- current step\nIn the `Contacts` app, create a new contact with the name "agent prog" and save it.  # <-- current step\nIn the `Contacts` app, create a new contact with the name "agent prog" and save it.  # <-- current step\nIn the `Contacts` app, create a new contact with the name "agent prog" and save it.  # <-- current step\n', 
        'code_context': '# Program Start: \n\n# Workflow step: In the `Contacts` app, create a new contact with the name "agent prog" and save it.\n\n# Next lines to generate', 
        'data_and_variables': "\nglobal variables:\nllm: <FoundationModel>\nmobile: <MobileAPI(config=MobileAPIConfig(locator='ui_tars', device_serial_id='emulator-5554', llm=<agentprog.all_ut...))>\ncurrent_screenshot: '{|{|/mnt/nvme0/home/mobile/AgentProg/outputs/images/screenshot_0.png|}|}'\n", 
        'belief_state': 'test', 
        'plan': 'Open the "Contacts" app.', 
        'action': 'mobile.start_app(app_name="Contacts")'
    }
    
    viz.show(sample_show_data, folded=False)
