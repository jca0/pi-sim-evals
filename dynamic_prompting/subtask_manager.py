"""
Breaks the task into ordered subtasks and advances through them based on progress monitor feedback.
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
class SubtaskPlan:
    high_level_instruction: str
    subtasks: list[str]


def decompose_task(instruction: str, model_id: str = "gemini-2.5-flash") -> SubtaskPlan:
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
        subtasks=subtask_strs,
    )


class SubtaskManager:
    """
    Tracks which subtask is currently active and handles transitions.
    """

    def __init__(self, plan: SubtaskPlan):
        self.plan = plan
        self._current_index = 0

    def reset(self):
        self._current_index = 0

    def current_instruction(self) -> str:
        """The instruction to send to the VLA right now."""
        if self.is_done():
            return self.plan.high_level_instruction
        return self.plan.subtasks[self._current_index]

    def total_subtasks(self) -> int:
        return len(self.plan.subtasks)

    def advance(self):
        """Move to the next subtask."""
        self._current_index += 1

    def is_done(self) -> bool:
        return self._current_index >= len(self.plan.subtasks)

    def status(self) -> str:
        if self.is_done():
            return f"All {self.total_subtasks()} subtasks completed"
        return (
            f"Subtask {self._current_index + 1}/{self.total_subtasks()}: "
            f'"{self.current_instruction()}"'
        )
