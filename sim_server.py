from pathlib import Path
import tempfile
import traceback
import subprocess

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI(title="DROID Simulator API")

ALLOWED_POLICIES = {"pi0.5", "pi0"}
ALLOWED_SCENES = {1, 2, 3, 4, 5, 6}

class SimRequest(BaseModel):
    instruction: str
    episode_length: float
    scene: int
    policy: str

def iterfile(path: Path): # what does this do lmao
    with open(path, "rb") as file:
        while chunk := file.read(1024 * 1024):
            yield chunk

@app.post("/simulate")
def simulator(request: SimRequest):
    # client side validation
    if request.policy not in ALLOWED_POLICIES:
        raise HTTPException(status_code=400, detail=f"Invalid policy: {request.policy}")
    if request.scene not in ALLOWED_SCENES:
        raise HTTPException(status_code=400, detail=f"Invalid scene: {request.scene}")

    tmp_root = Path(tempfile.mkdtemp(prefix="sim_run_"))

    cmd = [
        "python3",
        "run_eval.py",
        "--episodes", "1",
        "--episode_length", str(request.episode_length),
        "--headless",
        "--instruction", request.instruction,
        "--scene", str(request.scene),
        "--policy", request.policy,
        "--output_dir", str(tmp_root),
    ]

    result = subprocess.run(cmd,
                capture_output=True,
                text=True)

    if result.returncode != 0:
        print("run_eval stdout:\n", result.stdout)
        print("run_eval stderr:\n", result.stderr)
        raise HTTPException(status_code=500, detail=f"Simulation failed: {result.stderr}")

    mp4_files = sorted(tmp_root.glob("*.mp4"))
    if not mp4_files:
        raise HTTPException(status_code=500, detail="Simulation produced no video")
    
    video_path = mp4_files[-1]
    print("serving video: ", video_path.name)

    return StreamingResponse(
        iterfile(video_path),
        media_type="video/mp4",
        headers={
            "Content-Disposition": f'inline; filename=\"{video_path.name}\"'
        }
    )
