"""Tiptop websocket client - connects to tiptop server, gets a plan, executes it."""

import logging
import time
from typing import Optional

import msgpack_numpy
import numpy as np
import websockets.sync.client
from scipy.spatial.transform import Rotation

from .abstract_client import InferenceClient

try:
    from openpi_client import image_tools
except ImportError:
    image_tools = None

_log = logging.getLogger(__name__)
msgpack_numpy.patch()


class TiptopWebsocketClient(InferenceClient):
    """Queries tiptop server once for a plan, then steps through it."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        gripper_action_steps: int = 20,
        sim_control_hz: float = 15.0,
        curobo_interp_hz: float = 50.0,
    ) -> None:
        self._uri = f"ws://{host}:{port}"
        self._gripper_action_steps = gripper_action_steps
        
        # Compute waypoint stride to match sim control rate with CuRobo's interpolation rate
        # CuRobo default is ~50 Hz, sim runs at 15 Hz, so we skip ~3 waypoints per step
        self._waypoint_stride = max(1, int(round(curobo_interp_hz / sim_control_hz)))
        _log.info(f"Waypoint stride: {self._waypoint_stride} (curobo={curobo_interp_hz}Hz, sim={sim_control_hz}Hz)")

        self._ws: Optional[websockets.sync.client.ClientConnection] = None
        self._server_metadata: dict = {}

        # Plan execution state
        self._plan: Optional[list] = None
        self._current_plan_step: int = 0
        self._current_trajectory: Optional[np.ndarray] = None
        self._current_waypoint_idx: int = 0
        self._gripper_action_pending: Optional[str] = None
        self._gripper_action_steps_remaining: int = 0
        self._last_gripper_state: float = 0.0

        self._connect()

    def _connect(self) -> None:
        _log.info(f"Connecting to tiptop server at {self._uri}...")
        while True:
            try:
                self._ws = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None
                )
                raw_metadata = self._ws.recv()
                self._server_metadata = msgpack_numpy.unpackb(raw_metadata)
                _log.info(f"Connected to tiptop server: {self._server_metadata}")
                break
            except ConnectionRefusedError:
                _log.info("Waiting for tiptop server...")
                time.sleep(5)

    def reset(self) -> None:
        self._plan = None
        self._current_plan_step = 0
        self._current_trajectory = None
        self._current_waypoint_idx = 0
        self._gripper_action_pending = None
        self._gripper_action_steps_remaining = 0
        self._last_gripper_state = 0.0

    def infer(self, obs: dict, instruction: str) -> dict:
        curr_obs = self._extract_observation(obs)

        if self._plan is None:
            self._query_server(obs, curr_obs, instruction)

        return self._step_plan(curr_obs)

    def _query_server(self, raw_obs: dict, curr_obs: dict, instruction: str) -> None:
        _log.info(f"Querying tiptop server for task: '{instruction}'")

        request = self._build_request(raw_obs, curr_obs, instruction)

        packer = msgpack_numpy.Packer()
        self._ws.send(packer.pack(request))

        _log.info("Waiting for plan from tiptop server (this may take a while)...")
        start_time = time.time()
        response = msgpack_numpy.unpackb(self._ws.recv())
        elapsed = time.time() - start_time

        if response["success"]:
            self._plan = response["plan"]
            _log.info(f"Received plan with {len(self._plan)} steps in {elapsed:.1f}s")
            for i, step in enumerate(self._plan):
                if step["type"] == "trajectory":
                    orig_len = len(step["positions"])
                    subsampled_len = len(self._subsample_trajectory(step["positions"]))
                    _log.info(
                        f"  Step {i}: trajectory ({orig_len} -> {subsampled_len} waypoints after subsampling)"
                    )
                else:
                    _log.info(f"  Step {i}: gripper {step['action']}")
        else:
            _log.error(f"Tiptop server returned error: {response.get('error', 'unknown')}")
            self._plan = []

    def _build_request(self, raw_obs: dict, curr_obs: dict, instruction: str) -> dict:
        wrist_rgb = curr_obs["wrist_image"]
        wrist_depth = self._get_wrist_depth(raw_obs)
        intrinsics, world_from_cam = self._get_camera_params(raw_obs)
        q_init = curr_obs["joint_position"].flatten().astype(np.float32)

        return {
            "rgb": wrist_rgb.astype(np.uint8),
            "depth": wrist_depth.astype(np.float32),
            "intrinsics": intrinsics.astype(np.float32),
            "world_from_cam": world_from_cam.astype(np.float32),
            "task": instruction,
            "q_init": q_init,
        }

    def _get_wrist_depth(self, raw_obs: dict) -> np.ndarray:
        policy = raw_obs.get("policy", {})

        if "wrist_depth" in policy:
            depth = policy["wrist_depth"][0]
            if hasattr(depth, 'cpu'):
                depth = depth.cpu().numpy()
        else:
            raise ValueError("Wrist camera depth not found. Add 'wrist_depth' observation term.")

        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth.squeeze(-1)

        return depth

    def _get_camera_params(self, raw_obs: dict) -> tuple[np.ndarray, np.ndarray]:
        policy = raw_obs.get("policy", {})

        if "wrist_intrinsics" in policy:
            intrinsics = policy["wrist_intrinsics"][0]
            if hasattr(intrinsics, 'cpu'):
                intrinsics = intrinsics.cpu().numpy()
        else:
            # Default intrinsics for 1280x720, focal_length=2.8, horizontal_aperture=5.376
            fx = 1280 * 2.8 / 5.376
            intrinsics = np.array([[fx, 0, 640], [0, fx, 360], [0, 0, 1]], dtype=np.float32)
            _log.warning("Using default camera intrinsics")

        if "wrist_cam_pos_w" in policy and "wrist_cam_quat_w" in policy:
            pos = policy["wrist_cam_pos_w"][0]
            quat = policy["wrist_cam_quat_w"][0]
            if hasattr(pos, 'cpu'):
                pos = pos.cpu().numpy()
            if hasattr(quat, 'cpu'):
                quat = quat.cpu().numpy()
            world_from_cam = self._pose_to_matrix(pos, quat)
        else:
            _log.warning("Camera extrinsics not found, using identity transform")
            world_from_cam = np.eye(4, dtype=np.float32)

        # offset point cloud to match TipTop grasp_frame (relative to gripper base).
        world_from_cam = world_from_cam.copy()
        world_from_cam[:3, 3] -= np.array([0.0, 0.0, 0.015], dtype=np.float32)

        return intrinsics, world_from_cam

    def _pose_to_matrix(self, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
        """Convert pos + quat (wxyz) to 4x4 matrix."""
        quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float32)
        R = Rotation.from_quat(quat_xyzw).as_matrix()
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = pos
        return T

    def _subsample_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """Subsample waypoints while always keeping the final pose."""
        if self._waypoint_stride <= 1 or len(trajectory) == 0:
            return trajectory
        indices = np.arange(0, len(trajectory), self._waypoint_stride)
        if indices[-1] != len(trajectory) - 1:
            indices = np.append(indices, len(trajectory) - 1)
        return trajectory[indices]

    def _step_plan(self, curr_obs: dict) -> dict:
        # Handle pending gripper action
        if self._gripper_action_pending is not None:
            if self._gripper_action_steps_remaining > 0:
                self._gripper_action_steps_remaining -= 1
                joint_pos = curr_obs["joint_position"]
                gripper_val = 1.0 if self._gripper_action_pending == "close" else 0.0
                self._last_gripper_state = gripper_val
                action = np.concatenate([joint_pos.flatten(), np.array([gripper_val])])
                return self._make_result(action, curr_obs)
            else:
                self._last_gripper_state = 1.0 if self._gripper_action_pending == "close" else 0.0
                self._gripper_action_pending = None
                self._current_plan_step += 1

        # Load next trajectory if needed
        if self._current_trajectory is None or self._current_waypoint_idx >= len(self._current_trajectory):
            # Plan completed - hold position
            if self._plan is None or self._current_plan_step >= len(self._plan):
                joint_pos = curr_obs["joint_position"]
                gripper_val = curr_obs["gripper_position"][0] if len(curr_obs["gripper_position"]) > 0 else self._last_gripper_state
                action = np.concatenate([joint_pos.flatten(), np.array([gripper_val])])
                return self._make_result(action, curr_obs)

            step = self._plan[self._current_plan_step]

            if step["type"] == "gripper":
                self._gripper_action_pending = step["action"]
                self._gripper_action_steps_remaining = self._gripper_action_steps
                joint_pos = curr_obs["joint_position"]
                gripper_val = 1.0 if self._gripper_action_pending == "close" else 0.0
                self._last_gripper_state = gripper_val
                action = np.concatenate([joint_pos.flatten(), np.array([gripper_val])])
                return self._make_result(action, curr_obs)
            else:
                # Subsample trajectory to match sim control rate
                full_trajectory = step["positions"]
                self._current_trajectory = self._subsample_trajectory(full_trajectory)
                _log.debug(
                    f"Subsampled trajectory: {len(full_trajectory)} -> {len(self._current_trajectory)} waypoints"
                )
                self._current_waypoint_idx = 0
                self._current_plan_step += 1

        # Return next waypoint
        waypoint = self._current_trajectory[self._current_waypoint_idx]
        self._current_waypoint_idx += 1

        if waypoint.shape[0] == 7:
            action = np.concatenate([waypoint, np.array([self._last_gripper_state])])
        else:
            action = waypoint

        return self._make_result(action, curr_obs)

    def _make_result(self, action: np.ndarray, curr_obs: dict) -> dict:
        if image_tools is not None:
            img1 = image_tools.resize_with_pad(curr_obs["right_image"], 224, 224)
            img2 = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
            viz = np.concatenate([img1, img2], axis=1)
        else:
            viz = curr_obs["wrist_image"]

        return {
            "action": action,
            "viz": viz,
            "right_image": curr_obs["right_image"],
            "wrist_image": curr_obs["wrist_image"],
        }

    def _extract_observation(self, obs_dict: dict) -> dict:
        policy = obs_dict["policy"]
        return {
            "right_image": policy["external_cam"][0].clone().detach().cpu().numpy(),
            "wrist_image": policy["wrist_cam"][0].clone().detach().cpu().numpy(),
            "joint_position": policy["arm_joint_pos"].clone().detach().cpu().numpy(),
            "gripper_position": policy["gripper_pos"].clone().detach().cpu().numpy(),
        }

    def close(self) -> None:
        if self._ws is not None:
            self._ws.close()
            self._ws = None
