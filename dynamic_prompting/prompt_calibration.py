"""
Prompt calibration: generate prompt variations, judge rollout success,
and maintain a log of what worked and what didn't for a given task.
"""

import os
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

MODEL_ID = "gemini-2.5-flash"

VARIATION_PROMPT = """You are helping calibrate a robot Vision-Language-Action model. The robot needs to perform this task:

"{instruction}"

Generate {n} different ways to phrase this instruction. Vary:
- Word choice (e.g. "pick up" vs "grab" vs "grasp" vs "lift")
- Level of detail (e.g. "put cube in bowl" vs "carefully place the red cube into the center of the bowl")
- Sentence structure (e.g. imperative vs descriptive)

Each variation should mean the same thing but be phrased differently. These will be tested as prompts for a robot policy to find which phrasings work best.

Respond with JSON only, no markdown:
{{"variations": ["variation 1", "variation 2", ...]}}"""

SUCCESS_PROMPT = """You are a robot task completion judge. You are given an image showing the final state of a robot's workspace after it attempted a task.

The task was: "{instruction}"

Based on this image, determine whether the task was completed successfully.

Respond with JSON only, no markdown:
{{"success": true/false, "reason": "brief explanation"}}"""


def generate_prompt_variations(
    instruction: str,
    n: int = 5,
    model_id: str = MODEL_ID,
) -> list[str]:
    """
    Generate N phrasings of the same instruction using Gemini.

    Returns a list of instruction strings (does NOT include the original).
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment")

    client = genai.Client(api_key=api_key)
    prompt = VARIATION_PROMPT.format(instruction=instruction, n=n)

    response = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=1.0),
    )

    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]

    result = json.loads(text)
    return result["variations"]


def judge_success(
    frame: np.ndarray,
    instruction: str,
    model_id: str = MODEL_ID,
) -> dict:
    """
    Use Gemini to judge whether a task was completed from the final frame.

    Args:
        frame: Final camera frame (H, W, 3 uint8 RGB).
        instruction: The high-level task the robot was attempting.

    Returns:
        {"success": bool, "reason": str}
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment")

    client = genai.Client(api_key=api_key)
    prompt = SUCCESS_PROMPT.format(instruction=instruction)

    img = Image.fromarray(frame)
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    img_bytes = buf.getvalue()
    image_part = types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")

    response = client.models.generate_content(
        model=model_id,
        contents=[image_part, prompt],
        config=types.GenerateContentConfig(temperature=0.0),
    )

    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    try:
        result = json.loads(text)
        return {
            "success": bool(result.get("success", False)),
            "reason": str(result.get("reason", "")),
        }
    except json.JSONDecodeError:
        return {"success": False, "reason": f"Failed to parse VLM response: {text}"}


class CalibrationLog:
    """
    Stores and persists calibration results: which prompts/decompositions
    worked and which didn't for a given task.

    Results are saved as a JSON file that grows over time.
    """

    def __init__(self, log_dir: str = "calibration_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _log_path(self, scene: int) -> Path:
        return self.log_dir / f"scene_{scene}.json"

    def load(self, scene: int) -> list[dict]:
        path = self._log_path(scene)
        if path.exists():
            return json.loads(path.read_text())
        return []

    def save_result(
        self,
        scene: int,
        instruction: str,
        subtasks: list[str],
        success: bool,
        reason: str,
        video_path: str | None = None,
    ):
        """Append a single rollout result to the log."""
        entries = self.load(scene)
        entries.append({
            "timestamp": datetime.now().isoformat(),
            "instruction": instruction,
            "subtasks": subtasks,
            "success": success,
            "reason": reason,
            "video_path": video_path,
        })
        self._log_path(scene).write_text(json.dumps(entries, indent=2))

    def get_successful_prompts(self, scene: int) -> list[dict]:
        return [e for e in self.load(scene) if e["success"]]

    def get_failed_prompts(self, scene: int) -> list[dict]:
        return [e for e in self.load(scene) if not e["success"]]

    def summary(self, scene: int) -> str:
        entries = self.load(scene)
        if not entries:
            return f"Scene {scene}: no calibration data"
        successes = sum(1 for e in entries if e["success"])
        total = len(entries)
        lines = [f"Scene {scene}: {successes}/{total} successful"]
        for e in entries:
            status = "OK" if e["success"] else "FAIL"
            lines.append(f'  [{status}] "{e["instruction"]}" -> {e["subtasks"]}')
        return "\n".join(lines)
