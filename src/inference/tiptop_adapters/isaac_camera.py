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
    def __init__(self, camera):
        """
        Args:
            camera: the camera object from the environment
        """
        self.camera = camera
        # self.instrinsics_matrix = camera.data.intrinsic_matrices[0].cpu().numpy()
        # self.pos_w = camera.data.pos_w[0].cpu().numpy()
        # self.quat_w_ros = camera.data.quat_w_ros[0].cpu().numpy()
        # self.depth = camera.data.output["distance_to_image_plane"][0]

    def get_intrinsics(self):
        return self.camera.data.intrinsic_matrices[0].cpu().numpy()

    def get_pos_w(self):
        return self.camera.data.pos_w[0].cpu().numpy()

    def get_quat_w_ros(self):
        return self.camera.data.quat_w_ros[0].cpu().numpy()

    def get_depth(self):
        return self.camera.data.output["distance_to_image_plane"][0]

    def read_camera(self):
        rgb = self.camera.data.output["rgb"][0]
        depth = self.camera.data.output["distance_to_image_plane"][0]
        return IsaacSimFrame(rgb=rgb, depth=depth)
    