from dataclasses import dataclass

@dataclass
class BeliefState:
    belief_state_str: str
    plan: str