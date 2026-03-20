"""
Manages a sequence of subtasks for dynamic VLA prompting.

Instead of giving the VLA a single high-level instruction for the entire
episode, this module breaks the task into ordered subtasks and advances
through them based on progress monitor feedback.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

DECOMPOSITION_PROMPT = """You are a robot task planner. A robot arm needs to perform the following task:

"{instruction}"

Break this down into a short ordered list of atomic subtasks that a robot arm would execute sequentially. Each subtask should be a simple, single action (e.g. "pick up the cube", "move to the bowl", "place the cube in the bowl").

Keep it minimal — only include subtasks that are necessary. Typically 2-4 subtasks.

Respond with JSON only, no markdown:
{{"subtasks": ["subtask 1", "subtask 2", ...]}}"""


@dataclass
class Subtask:
    instruction: str
    max_steps: int | None = None  # optional per-subtask step limit


@dataclass
class SubtaskPlan:
    """An ordered sequence of subtasks that together accomplish a high-level task."""
    high_level_instruction: str
    subtasks: list[Subtask]


def decompose_task(instruction: str, model_id: str = "gemini-2.5-flash") -> SubtaskPlan:
    """
    Use Gemini to break a high-level instruction into atomic subtasks.

    Args:
        instruction: High-level task description (e.g. "put the cube in the bowl").
        model_id: Gemini model to use for decomposition.

    Returns:
        A SubtaskPlan with the generated subtasks.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment")

    client = genai.Client(api_key=api_key)
    prompt = DECOMPOSITION_PROMPT.format(instruction=instruction)

    response = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0),
    )

    text = response.text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]

    result = json.loads(text)
    subtask_strs = result["subtasks"]

    return SubtaskPlan(
        high_level_instruction=instruction,
        subtasks=[Subtask(instruction=s) for s in subtask_strs],
    )


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
