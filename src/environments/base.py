"""Abstract base classes for dataset environments."""

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class BaseState:
    question: str
    gold_answer: str
    example_id: str
    display_raw_actions: bool = False
    context: str = ""
    action_history: list[dict] = field(default_factory=list)
    done: bool = False
    step_count: int = 0
    submitted_answer: str | None = None


class BaseEnvironment(ABC):
    def __init__(self, example: dict):
        self.state = self._init_state(example)

    @abstractmethod
    def _init_state(self, example: dict) -> BaseState: ...

    @abstractmethod
    def step(self, action: str) -> str: ...

    def record_invalid_action(self, action: str, observation: str,
                              done: bool = False) -> str:
        """Record a non-executable action so the next prompt can recover.

        Invalid policy outputs should still become part of the trajectory:
        the agent sees the invalid-action observation on the next turn, and
        training/evaluation can penalize the behavior without silently dropping
        the step.
        """
        self.state.step_count += 1
        self.state.context += f"\n\nObservation: {observation}"
        self.state.action_history.append({"action": action, "observation": observation})
        if done:
            self.state.done = True
        return observation

    def get_state(self) -> BaseState:
        return copy.deepcopy(self.state)

    def set_state(self, state: BaseState) -> None:
        self.state = copy.deepcopy(state)

    @property
    def done(self) -> bool:
        return self.state.done
