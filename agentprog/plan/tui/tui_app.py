import json
import os
import sys
import asyncio
import dataclasses
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.syntax import Syntax
from rich.layout import Layout
from rich import box
import pyfiglet

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Header, Footer, Button, Input, Static, Label, Switch, Log
from textual.screen import Screen, ModalScreen
from textual.binding import Binding
from textual.worker import Worker, WorkerState
from textual.reactive import reactive
from textual.message import Message

from agentprog.plan.code_exec.workflow.config.core_config import AgentProgConfig

class ConfigManager:
    CONFIG_FILE = os.path.expanduser("~/.agentprog")

    @staticmethod
    def load_config() -> AgentProgConfig:
        defaults = AgentProgConfig.get_field_default_value()
        defaults['workflow_path'] = f"outputs/{datetime.now().strftime('%Y%m%d%H%M%S')}.ap"
        
        if os.path.exists(ConfigManager.CONFIG_FILE):
            try:
                with open(ConfigManager.CONFIG_FILE, 'r') as f:
                    saved_data = json.load(f)
                # Merge saved data into defaults
                for k, v in saved_data.items():
                    if k in defaults:
                        defaults[k] = v
            except Exception as e:
                pass # Fail silently and use defaults
        
        return AgentProgConfig(**defaults)

    @staticmethod
    def save_config(config: AgentProgConfig):
        data = dataclasses.asdict(config)
        try:
            with open(ConfigManager.CONFIG_FILE, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            pass

# ==========================================
# 2. Visualization Logic (Adapted for Textual)
# ==========================================


def get_title():
    return '''
 █████╗  ██████╗ ███████╗███╗   ██╗████████╗██████╗ ██████╗  ██████╗  ██████╗ 
██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝██╔══██╗██╔══██╗██╔═══██╗██╔════╝ 
███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   ██████╔╝██████╔╝██║   ██║██║  ███╗
██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   ██╔═══╝ ██╔══██╗██║   ██║██║   ██║
██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   ██║     ██║  ██║╚██████╔╝╚██████╔╝
╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝     ╚═╝  ╚═╝ ╚═════╝  ╚═════╝ 
'''

class DashboardLayoutRenderer:
    """Helper class to generate the Rich Layout for the dashboard."""
    def __init__(self):
        self.styles = {
            "step_id": ("bright_cyan", "👉"),
            "Observation": ("bright_green", "🔍"),
            "Thought": ("bright_yellow", "🧠"),
            "Updated_Belief_State": ("bright_blue", "📦"),
            "Judgement": ("bright_red", "⚖️"),
            "Plan": ("bright_magenta", "📋"),
            "Action": ("bright_white", "⚡"),
        }

    def _create_header(self):
        ascii_banner = get_title()
        return Panel(
            Text(ascii_banner, style="bold cyan", justify="center"),
            border_style="bright_blue",
            box=box.HEAVY,
            padding=(0, 1) 
        )

    def _get_content_renderable(self, key, content, color):
        icon = self.styles.get(key, ("white", ""))[1]
        title = f"{icon}  {key}"

        if isinstance(content, dict):
            json_str = json.dumps(content, indent=1, ensure_ascii=False)
            syntax = Syntax(json_str, "json", theme="monokai", line_numbers=False, word_wrap=True)
            return Panel(syntax, title=title, border_style=color, box=box.ROUNDED)
        else:
            txt_str = str(content)
            txt = Text(txt_str)
            txt.highlight_words(["CRITICAL_ERROR", "incorrect", "must be corrected"], style="bold red")
            return Panel(txt, title=title, border_style=color, box=box.ROUNDED)

    def create_layout(self, data: dict) -> Layout:
        layout = Layout()
        
        # Structure
        layout.split_column(
            Layout(name="header", size=6),
            Layout(name="main")
        )
        layout["header"].update(self._create_header())

        layout["main"].split_column(
            Layout(name="upper_body", ratio=3),
            Layout(name="footer", ratio=1)
        )

        layout["upper_body"].split_row(
            Layout(name="observation", ratio=1),
            Layout(name="reasoning", ratio=1)
        )
        
        layout["reasoning"].split_column(
            Layout(name="thought", ratio=1),
            Layout(name="belief", ratio=2)
        )

        layout["footer"].split_row(
            Layout(name="judgement"),
            Layout(name="plan"),
            Layout(name="action")
        )

        # Content
        def get_panel(k):
            color = self.styles.get(k, ("white", ""))[0]
            return self._get_content_renderable(k, data.get(k, ""), color)

        layout["observation"].update(get_panel("Observation"))
        layout["thought"].update(get_panel("Thought"))
        layout["belief"].update(get_panel("Updated_Belief_State"))
        layout["judgement"].update(get_panel("Judgement"))
        layout["plan"].update(get_panel("Plan"))
        layout["action"].update(get_panel("Action"))

        return layout

# ==========================================
# 3. Textual Application Screens
# ==========================================

class SettingsScreen(Screen):
    """Screen to configure AgentProg settings."""
    BINDINGS = [("escape", "app.pop_screen", "Back"), ("f10", "save", "Save & Back")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Label("[bold cyan]Configuration Settings[/bold cyan]", classes="title"),
            ScrollableContainer(id="settings-container"),
            Horizontal(
                Button("Save & Return", variant="success", id="btn-save"),
                Button("Cancel", variant="error", id="btn-cancel"),
                classes="buttons"
            ),
            classes="main-container"
        )
        yield Footer()

    def on_mount(self):
        container = self.query_one("#settings-container")
        config = self.app.config
        
        # Generate fields dynamically
        for field_name in AgentProgConfig.get_field_names():
            value = getattr(config, field_name)
            
            # Skip path fields that are auto-generated mostly
            if field_name in ["task_description", "workflow_path"]:
                continue

            container.mount(Label(f"{field_name.replace('_', ' ').title()}:"))
            
            if isinstance(value, bool):
                sw = Switch(value=value, id=f"field_{field_name}")
                container.mount(sw)
            else:
                val_str = str(value) if value is not None else ""
                inp = Input(value=val_str, id=f"field_{field_name}", placeholder=f"Enter {field_name}")
                container.mount(inp)

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "btn-save":
            self.save_settings()
            self.app.pop_screen()
        elif event.button.id == "btn-cancel":
            self.app.pop_screen()

    def action_save(self):
        self.save_settings()
        self.app.pop_screen()

    def save_settings(self):
        config = self.app.config
        for field_name in AgentProgConfig.get_field_names():
            if field_name in ["task_description", "workflow_path"]:
                continue
                
            widget_id = f"#field_{field_name}"
            try:
                widget = self.query_one(widget_id)
                if isinstance(widget, Switch):
                    setattr(config, field_name, widget.value)
                elif isinstance(widget, Input):
                    val = widget.value
                    # Simple type conversion logic
                    if val.lower() == "none" or val == "":
                        setattr(config, field_name, None)
                    else:
                        setattr(config, field_name, val)
            except:
                pass
        
        ConfigManager.save_config(config)
        self.notify("Configuration Saved!")


class CodeReviewScreen(Screen):
    """Screen to review generated task code."""
    def __init__(self, generated_code: str):
        super().__init__()
        self.generated_code = generated_code

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Label("[bold yellow]Generated Task Program Review[/bold yellow]", classes="title"),
            Static(renderable=Syntax(self.generated_code, "python", theme="monokai", line_numbers=True), id="code-view"),
            Horizontal(
                Button("Execute Task", variant="success", id="btn-exec"),
                Button("Reject / Back", variant="error", id="btn-back"),
                classes="buttons"
            ),
            classes="main-container"
        )
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "btn-exec":
            self.app.push_screen(ExecutionScreen())
        elif event.button.id == "btn-back":
            self.app.pop_screen()


class DashboardWidget(Static):
    """Widget to hold the Rich Layout."""
    def __init__(self):
        super().__init__()
        self.renderer = DashboardLayoutRenderer()
        self.current_data = {}
    
    def update_data(self, data):
        self.current_data = data
        self.refresh()

    def render(self):
        # If no data yet, show a placeholder
        if not self.current_data:
            return Panel("Waiting for Agent execution...", title="AgentProg", style="dim")
        return self.renderer.create_layout(self.current_data)


class ExecutionScreen(Screen):
    """Screen displaying the execution dashboard."""
    BINDINGS = [("ctrl+c", "interrupt_task", "Interrupt Task")]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield DashboardWidget()
        yield Footer()

    def on_mount(self):
        self.worker = self.run_worker(self.run_task_thread, exclusive=True)

    async def action_interrupt_task(self):
        self.notify("Interrupting Task...", severity="warning")
        if self.worker:
            self.worker.cancel()
        await asyncio.sleep(0.5) # Give it a moment
        self.app.pop_screen() # Go back to previous screen (Code Review)
        # Or go back to Main:
        # self.app.pop_screen()
        # self.app.pop_screen() 

    async def run_task_thread(self):
        dashboard = self.query_one(DashboardWidget)
        config = self.app.config
        
        # =======================================================
        # TODO: HERE YOU WOULD CALL THE REAL AGENTPROG PIPELINE
        # workflow_result = agentprog_pipeline_core(config)
        # But since we need to visualize steps, the pipeline 
        # needs to yield data or use a callback.
        # =======================================================
        
        # Mocking the pipeline execution for demonstration
        await self.run_pipeline_simulation(dashboard)

    async def run_pipeline_simulation(self, dashboard_widget):
        """Simulates the agent receiving updates."""
        steps = [
             {
                "step_id": 0,
                "Observation": "Initializing...",
                "Thought": "Preparing environment.",
            },
            {
                "step_id": 1,
                "Observation": 'The current screen is the "New Recipe" page.',
                "Thought": "I see incorrect text in Title.",
                "Updated_Belief_State": {"status": "Checking fields"},
                "Judgement": "Need to correct title.",
                "Plan": "Locate Title field and clear it.",
                "Action": "ui.locate('Title')"
            },
             {
                "step_id": 2,
                "Observation": 'Title field focused.',
                "Thought": "Now I will type the correct title.",
                "Updated_Belief_State": {
                    "current_page": "New Recipe",
                    "CRITICAL_ERROR": 'Title was wrong',
                    "Description": "filled correctly",
                },
                "Judgement": "Proceed with correction.",
                "Plan": 'Input "Chicken Caesar Salad Wrap".',
                "Action": 'ui.input("Chicken Caesar Salad Wrap")'
            },
            {
                "step_id": 3,
                "Observation": 'Title updated correctly.',
                "Thought": "Task looks good. Verification complete.",
                "Updated_Belief_State": {"task_complete": True},
                "Judgement": "Success.",
                "Plan": "Finish task.",
                "Action": "DONE"
            }
        ]

        for step_data in steps:
            dashboard_widget.update_data(step_data)
            # Simulate processing time
            await asyncio.sleep(2)

        self.notify("Task Execution Completed!", severity="information")


class MainScreen(Screen):
    """The entry point screen."""
    BINDINGS = [("escape", "app.quit", "Quit App")]
    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static(self.get_banner(), id="banner"),
            Label("Enter Task Description:", classes="label"),
            Input(placeholder="e.g. Open settings and toggle Wi-Fi", id="task-input"),
            Horizontal(
                Button("Generate & Run", variant="primary", id="btn-run"),
                Button("Settings", variant="default", id="btn-settings"),
                classes="buttons"
            ),
            classes="main-container"
        )
        yield Footer()

    def get_banner(self):
        return Text(get_title(), style="orange")

    def on_mount(self):
        self.query_one("#task-input").focus()

    async def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "btn-settings":
            self.app.push_screen(SettingsScreen())
        
        elif event.button.id == "btn-run":
            task_desc = self.query_one("#task-input").value
            if not task_desc.strip():
                self.notify("Please enter a task description.", severity="error")
                return
            
            # Update config with current task
            self.app.config.task_description = task_desc
            
            # Check Serial / ADB
            if not self.app.config.serial:
                self.notify("ADB Serial not configured! Please set it in Settings.", severity="warning")
                # Optionally direct them to a mini-prompt or Settings
                await self.prompt_for_serial()
            else:
                await self.start_generation_process()

    async def prompt_for_serial(self):
        # A simple way to ask for serial if missing
        def check_serial(text):
            if text:
                self.app.config.serial = text
                ConfigManager.save_config(self.app.config)
                self.run_worker(self.start_generation_process())

        self.app.push_screen(InputModal(title="Enter ADB Serial (e.g. emulator-5554):"), check_serial)

    async def start_generation_process(self):
        self.notify("Generating Task Program...", title="AgentProg")
        
        # Simulate code generation delay
        await asyncio.sleep(1) 
        
        # Mock generated code
        mock_code = f'''
# Generated Task: {self.app.config.task_description}
from agentprog.actions import AndroidUI

def run_task(ui: AndroidUI):
    """
    Task: {self.app.config.task_description}
    Device: {self.app.config.serial}
    """
    # 1. Open App
    ui.launch_app("com.example.settings")
    
    # 2. Locate item
    ui.wait_for_text("Network & internet")
    ui.click_text("Network & internet")
    
    # 3. Toggle
    ui.click_id("switch_widget")
    
    return True
'''
        self.app.push_screen(CodeReviewScreen(mock_code))


class InputModal(ModalScreen):
    """Simple modal to get a single input."""
    def __init__(self, title, callback):
        super().__init__()
        self.title_text = title
        self.callback = callback

    def compose(self) -> ComposeResult:
        yield Container(
            Label(self.title_text),
            Input(id="modal-input"),
            Button("Confirm", id="modal-confirm"),
            classes="modal-box"
        )

    def on_button_pressed(self, event: Button.Pressed):
        val = self.query_one("#modal-input").value
        self.dismiss()
        self.callback(val)

# ==========================================
# 4. Main Application Class & CSS
# ==========================================

class AgentProgApp(App):
    CSS = """
    Screen {
        align: center middle;
    }

    .main-container {
        width: 80%;
        height: 90%;
        border: thick $accent;
        padding: 1 2;
        background: $surface;
    }

    .title {
        text-align: center;
        padding-bottom: 1;
    }

    #banner {
        content-align: center middle;
        height: auto;
        margin-bottom: 1;
    }

    .buttons {
        align: center middle;
        height: auto;
        margin-top: 1;
    }
    
    Button {
        margin: 0 1;
    }

    #settings-container {
        height: 1fr;
        border: solid $secondary;
        scrollbar-gutter: stable;
        padding: 0 1;
    }
    
    #code-view {
        height: 100%;
        border: solid $secondary;
    }

    /* Modal Styles */
    .modal-box {
        width: 50%;
        height: auto;
        background: $panel;
        border: heavy $accent;
        padding: 2;
        align: center middle;
    }
    
    Label {
        margin-bottom: 1;
    }
    
    Switch {
        margin-bottom: 1;
    }
    """

    def __init__(self):
        super().__init__()
        self.config = ConfigManager.load_config()

    def on_mount(self):
        self.push_screen(MainScreen())

# ==========================================
# 5. Entry Point
# ==========================================

def agentprog_interaction_cli():
    app = AgentProgApp()
    app.run()

if __name__ == "__main__":
    agentprog_interaction_cli()
