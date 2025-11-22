from pathlib import Path
import tempfile
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from run_eval import run_simulation

app = FastAPI(title="DROID Simulator API")

ALLOWED_POLICIES = {"pi0.5", "pi0"}
ALLOWED_SCENES = {1, 2, 3, 4, 5, 6}

class SimRequest(BaseModel):
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

    try:
        video_dir = run_simulation(
            episodes=1,
            headless=True,
            scene=request.scene,
            policy=request.policy,
            output_dir=tmp_root,
        )

        mp4_files = sorted(video_dir.glob("*.mp4"))
        if not mp4_files:
            raise HTTPException(status_code=500, detail="Simulation produced no video")
        
        video_path = mp4_files[-1]
        print(video_path.name)

        return StreamingResponse(
            iterfile(video_path),
            media_type="video/mp4",
            headers={
                "Content-Disposition": f'inline; filename="{video_path.name}"'
            }
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Simulation failed: {e}")