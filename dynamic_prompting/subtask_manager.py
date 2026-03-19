"""
Manages a sequence of subtasks for dynamic VLA prompting.

Instead of giving the VLA a single high-level instruction for the entire
episode, this module breaks the task into ordered subtasks and advances
through them based on progress monitor feedback.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Subtask:
    instruction: str
    max_steps: int | None = None  # optional per-subtask step limit


@dataclass
class SubtaskPlan:
    """An ordered sequence of subtasks that together accomplish a high-level task."""
    high_level_instruction: str
    subtasks: list[Subtask]


class SubtaskManager:
    """
    Tracks which subtask is currently active and handles transitions.

    Usage:
        manager = SubtaskManager(plan)
        while not manager.is_done():
            instruction = manager.current_instruction()
            # ... run VLA with instruction ...
            if progress_monitor says done:
                manager.advance()
    """

    def __init__(self, plan: SubtaskPlan):
        self.plan = plan
        self._current_index = 0
        self._subtask_step_count = 0

    def reset(self):
        self._current_index = 0
        self._subtask_step_count = 0

    def current_instruction(self) -> str:
        """The instruction to send to the VLA right now."""
        if self.is_done():
            return self.plan.high_level_instruction
        return self.plan.subtasks[self._current_index].instruction

    def current_subtask(self) -> Subtask | None:
        if self.is_done():
            return None
        return self.plan.subtasks[self._current_index]

    def current_index(self) -> int:
        return self._current_index

    def total_subtasks(self) -> int:
        return len(self.plan.subtasks)

    def advance(self):
        """Move to the next subtask."""
        self._current_index += 1
        self._subtask_step_count = 0

    def step(self):
        """Call once per env step to track per-subtask step counts."""
        self._subtask_step_count += 1

    def is_done(self) -> bool:
        return self._current_index >= len(self.plan.subtasks)

    def exceeded_subtask_limit(self) -> bool:
        """True if the current subtask has a step limit and we've exceeded it."""
        subtask = self.current_subtask()
        if subtask is None or subtask.max_steps is None:
            return False
        return self._subtask_step_count >= subtask.max_steps

    def status(self) -> str:
        if self.is_done():
            return f"All {self.total_subtasks()} subtasks completed"
        return (
            f"Subtask {self._current_index + 1}/{self.total_subtasks()}: "
            f'"{self.current_instruction()}" '
            f"(step {self._subtask_step_count})"
        )


# ── Pre-defined task plans ──────────────────────────────────────────────
# Add or modify these to match your scenes.
# Each plan maps a high-level instruction to an ordered list of subtasks.

TASK_PLANS: dict[int, SubtaskPlan] = {
    1: SubtaskPlan(
        high_level_instruction="put the cube in the bowl",
        subtasks=[
            Subtask("pick up the cube"),
            Subtask("place the cube in the bowl"),
        ],
    ),
    2: SubtaskPlan(
        high_level_instruction="pick up the can and put it in the mug",
        subtasks=[
            Subtask("pick up the can"),
            Subtask("place the can in the mug"),
        ],
    ),
    3: SubtaskPlan(
        high_level_instruction="put the banana in the bin",
        subtasks=[
            Subtask("pick up the banana"),
            Subtask("place the banana in the bin"),
        ],
    ),
    4: SubtaskPlan(
        high_level_instruction="put the mug on top of the sugar box",
        subtasks=[
            Subtask("pick up the mug"),
            Subtask("place the mug on top of the sugar box"),
        ],
    ),
    5: SubtaskPlan(
        high_level_instruction="grasp a cube",
        subtasks=[
            Subtask("grasp a cube"),
        ],
    ),
}
