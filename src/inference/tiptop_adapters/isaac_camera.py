import numpy as np
from dataclasses import dataclass
from jaxtyping import Float, UInt8

@dataclass
class IsaacSimFrame:
    timestamp: float # probably don't need?
    rgb: UInt8[np.ndarray, "h w 3"]
    depth: Float[np.ndarray, "h w"] | None

@dataclass
class IsaacSimIntrinsics:
    K: Float[np.ndarray, "3 3"]

@dataclass
class IsaacSimExtrinsics:
    R: Float[np.ndarray, "3 3"]
    T: Float[np.ndarray, "3"]

class IsaacSimCamera:
    def __init__(self, intrinsics: np.ndarray, extrinsics: np.ndarray, frame_extractor):
        # frame extractor is a function that takes in a dictionary of observations and returns a tuple of (rgb, depth)
        self._intrinsics = IsaacSimIntrinsics(K=np.asarray(intrinsics, dtype=np.float32))
        self._extrinsics = IsaacSimExtrinsics(R=np.asarray(extrinsics, dtype=np.float32), T=np.asarray(extrinsics, dtype=np.float32))
        self._frame_extractor = frame_extractor
        self.serial = "isaac_sim_cam"

    def get_intrinsics(self) -> IsaacSimIntrinsics:
        return self._intrinsics

    def get_extrinsics(self) -> IsaacSimExtrinsics:
        return self._extrinsics

    def read_camera(self, obs_dict) -> IsaacSimFrame:
        rgb, depth = self._frame_extractor(obs_dict)
        return IsaacSimFrame(rgb=rgb, depth=depth)
    