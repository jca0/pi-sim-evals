"""
Progress monitor that uses a VLM (Google Gemini) to check whether
a robot subtask has been completed, based on the current camera frame.
"""

import os
import json

import numpy as np
from PIL import Image
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

MODEL_ID = "gemini-2.5-flash"

COMPLETION_PROMPT_TEMPLATE = """You are a robot task completion checker. You are given a single frame from a robot's camera.

The robot's current subtask is: "{subtask}"

Based on this frame, determine whether the subtask has been completed.

Respond with JSON only, no markdown:
{{"completed": true/false, "reason": "brief explanation"}}"""


class ProgressMonitor:
    """Calls a VLM every N steps to check if the current subtask is done."""

    def __init__(
        self,
        check_every_n_steps: int = 15,
        model_id: str = MODEL_ID,
    ):
        """
        Args:
            check_every_n_steps: How often (in env steps) to query the VLM.
            model_id: Gemini model to use.
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")

        self.client = genai.Client(api_key=api_key)
        self.model_id = model_id
        self.check_every_n_steps = check_every_n_steps
        self._step_count = 0
        self._latest_frame: np.ndarray | None = None

    def reset(self):
        self._step_count = 0
        self._latest_frame = None

    def set_frame(self, frame: np.ndarray):
        """Store the current camera frame (H, W, 3 uint8 RGB)."""
        self._latest_frame = frame

    def should_check(self) -> bool:
        """Returns True if it's time to query the VLM."""
        self._step_count += 1
        return (
            self._step_count % self.check_every_n_steps == 0
            and self._latest_frame is not None
        )

    def check_completion(self, subtask: str) -> dict:
        """
        Query the VLM to determine if the subtask is complete.

        Args:
            subtask: Natural language description of the current subtask.

        Returns:
            {"completed": bool, "reason": str}
        """
        prompt = COMPLETION_PROMPT_TEMPLATE.format(subtask=subtask)

        img = Image.fromarray(self._latest_frame)
        img_bytes = _image_to_bytes(img)
        image_part = types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=[image_part, prompt],
            config=types.GenerateContentConfig(temperature=0.0),
        )

        return _parse_response(response.text)


def _image_to_bytes(img: Image.Image) -> bytes:
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _parse_response(text: str) -> dict:
    """Best-effort parse of the VLM JSON response."""
    text = text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    try:
        result = json.loads(text)
        return {
            "completed": bool(result.get("completed", False)),
            "reason": str(result.get("reason", "")),
        }
    except json.JSONDecodeError:
        return {"completed": False, "reason": f"Failed to parse VLM response: {text}"}
