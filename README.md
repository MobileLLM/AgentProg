# AgentProg: Empowering Long-Horizon GUI Agents with Program-Guided Context Management 


<p align="center">
| <a href="https://arxiv.org/pdf/"><b>Paper</b></a> |
</p>

## Setup

Setup agentprog package:
```
git clone https://github.com/MobileLLM/AgentProg.git
cd AgentProg
pip install -e .
```

python 3.11+ is recommended for running the agent.

Set Gemini api key and base url in `.env`:

```
GEMINI_API_KEY=<YOUR_API_KEY_HERE>
```

## Getting Started

For cli usage:
```
agent_prog [Task requirements] --serial [Serial name, e.g., emulator-5554]
```

For example:
```
agentprog "create a new contact named agent prog in contacts app." --serial emulator-5554
```

You can also use AgentProg in Python:

```python
from agentprog import agentprog_pipeline_core
config = AgentProgConfig(task_description="create a new contact named agent prog in contacts app.", serial="emulator-5554")
agentprog_pipeline_core(config)
```

## Citation
```bibtex

```