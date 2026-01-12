"""Tiptop websocket client for perception + planning.

This client connects to the tiptop websocket server, sends initial observations
(RGB, depth, camera intrinsics/extrinsics, task instruction, and initial joint positions),
receives a trajectory plan, and then steps through the plan in subsequent infer() calls.
"""

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

# Register msgpack_numpy for numpy array serialization
msgpack_numpy.patch()


class TiptopWebsocketClient(InferenceClient):
    """Client that queries tiptop server once for a plan, then steps through it.

    On the first call to infer(), this client:
    1. Extracts RGB, depth, camera intrinsics/extrinsics from the observation
    2. Sends them to the tiptop websocket server along with task instruction
    3. Receives a trajectory plan
    4. Caches the plan and steps through it on subsequent infer() calls

    The stepping logic mirrors cutamp_jointpos.py.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        gripper_action_steps: int = 20,
    ) -> None:
        """Initialize the tiptop websocket client.

        Args:
            host: Websocket server host
            port: Websocket server port
            gripper_action_steps: Number of steps to hold gripper action
        """
        self._uri = f"ws://{host}:{port}"
        self._gripper_action_steps = gripper_action_steps

        # Connection and metadata
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

        # Connect to server
        self._connect()

    def _connect(self) -> None:
        """Connect to the websocket server."""
        _log.info(f"Connecting to tiptop server at {self._uri}...")
        while True:
            try:
                self._ws = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None
                )
                # Receive server metadata
                raw_metadata = self._ws.recv()
                self._server_metadata = msgpack_numpy.unpackb(raw_metadata)
                _log.info(f"Connected to tiptop server: {self._server_metadata}")
                break
            except ConnectionRefusedError:
                _log.info("Waiting for tiptop server...")
                time.sleep(5)

    def reset(self) -> None:
        """Reset the client state for a new episode."""
        self._plan = None
        self._current_plan_step = 0
        self._current_trajectory = None
        self._current_waypoint_idx = 0
        self._gripper_action_pending = None
        self._gripper_action_steps_remaining = 0
        self._last_gripper_state = 0.0

    def infer(self, obs: dict, instruction: str) -> dict:
        """Infer the next action.

        On the first call, sends observation to the server and receives a plan.
        On subsequent calls, steps through the cached plan.

        Args:
            obs: Observation dictionary from the environment
            instruction: Task instruction string

        Returns:
            Dictionary with:
                - action: np.ndarray of shape (8,) - 7 joint positions + 1 gripper
                - viz: Visualization image
                - right_image: External camera image
                - wrist_image: Wrist camera image
        """
        curr_obs = self._extract_observation(obs)

        # If no plan yet, query the server
        if self._plan is None:
            self._query_server(obs, curr_obs, instruction)

        # Step through the plan
        return self._step_plan(curr_obs)

    def _query_server(self, raw_obs: dict, curr_obs: dict, instruction: str) -> None:
        """Query the tiptop server for a plan.

        Args:
            raw_obs: Raw observation dictionary from environment
            curr_obs: Extracted observation dictionary
            instruction: Task instruction
        """
        _log.info(f"Querying tiptop server for task: '{instruction}'")

        # Build the request
        request = self._build_request(raw_obs, curr_obs, instruction)

        # Send request and receive plan
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
                    n_waypoints = len(step["positions"])
                    _log.info(f"  Step {i}: trajectory ({n_waypoints} waypoints)")
                else:
                    _log.info(f"  Step {i}: gripper {step['action']}")
        else:
            _log.error(f"Tiptop server returned error: {response.get('error', 'unknown')}")
            # Use empty plan so we just hold position
            self._plan = []

    def _build_request(self, raw_obs: dict, curr_obs: dict, instruction: str) -> dict:
        """Build the request dictionary to send to the server.

        Args:
            raw_obs: Raw observation from environment (contains camera sensor data)
            curr_obs: Extracted observation dictionary
            instruction: Task instruction

        Returns:
            Request dictionary with RGB, depth, intrinsics, extrinsics, task, q_init
        """
        # Get wrist camera data - need RGB and depth
        policy = raw_obs["policy"]
        wrist_rgb = curr_obs["wrist_image"]  # Already extracted as numpy uint8

        # Get depth from the raw observation
        # In the environment, wrist_cam has distance_to_image_plane
        # We need to access the camera sensor directly for depth
        wrist_depth = self._get_wrist_depth(raw_obs)

        # Get camera intrinsics and extrinsics from the environment
        intrinsics, world_from_cam = self._get_camera_params(raw_obs)

        # Get current joint positions
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
        """Extract wrist camera depth from observation.

        Note: This requires accessing the camera sensor data.
        The depth is in the 'distance_to_image_plane' data type.

        Args:
            raw_obs: Raw observation dictionary

        Returns:
            Depth array (H, W) in meters
        """
        policy = raw_obs.get("policy", {})
        
        if "wrist_depth" in policy:
            depth = policy["wrist_depth"][0]
            if hasattr(depth, 'cpu'):
                depth = depth.cpu().numpy()
        else:
            raise ValueError(
                "Wrist camera depth not found in observation. "
                "Make sure the environment provides 'wrist_depth' observation term."
            )

        # Squeeze if needed
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth.squeeze(-1)

        return depth

    def _get_camera_params(self, raw_obs: dict) -> tuple[np.ndarray, np.ndarray]:
        """Get camera intrinsics and world_from_cam transform.

        Args:
            raw_obs: Raw observation dictionary

        Returns:
            Tuple of (intrinsics (3,3), world_from_cam (4,4))
        """
        policy = raw_obs.get("policy", {})
        
        # Get intrinsics from observation or use defaults
        if "wrist_intrinsics" in policy:
            intrinsics = policy["wrist_intrinsics"][0]
            if hasattr(intrinsics, 'cpu'):
                intrinsics = intrinsics.cpu().numpy()
        else:
            # Use default intrinsics based on camera config
            # From droid_environment.py: focal_length=2.8, horizontal_aperture=5.376
            # For 1280x720 resolution
            # fx = fy = width * focal_length / horizontal_aperture
            fx = 1280 * 2.8 / 5.376
            fy = fx  # Assuming square pixels
            cx = 1280 / 2
            cy = 720 / 2
            intrinsics = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ], dtype=np.float32)
            _log.warning("Using default camera intrinsics")

        # Get extrinsics from observation
        if "wrist_cam_pos_w" in policy and "wrist_cam_quat_w" in policy:
            pos = policy["wrist_cam_pos_w"][0]
            quat = policy["wrist_cam_quat_w"][0]
            if hasattr(pos, 'cpu'):
                pos = pos.cpu().numpy()
            if hasattr(quat, 'cpu'):
                quat = quat.cpu().numpy()
            world_from_cam = self._pose_to_matrix(pos, quat)
        else:
            # Fallback: identity (this should be updated)
            _log.warning("Camera extrinsics not found, using identity transform")
            world_from_cam = np.eye(4, dtype=np.float32)

        return intrinsics, world_from_cam

    def _pose_to_matrix(self, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
        """Convert position and quaternion to 4x4 transformation matrix.

        Args:
            pos: Position (3,)
            quat: Quaternion - assumed to be [x, y, z, w] (scipy convention)

        Returns:
            4x4 transformation matrix
        """
        R = Rotation.from_quat(quat).as_matrix()
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = pos
        return T

    def _step_plan(self, curr_obs: dict) -> dict:
        """Step through the cached plan.

        This logic mirrors cutamp_jointpos.py.

        Args:
            curr_obs: Extracted observation dictionary

        Returns:
            Action dictionary
        """
        # Handle gripper actions first if pending
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

        # Check if we need to load a new action from the plan
        if self._current_trajectory is None or self._current_waypoint_idx >= len(self._current_trajectory):
            # Check if we've completed all plan steps
            if self._plan is None or self._current_plan_step >= len(self._plan):
                # Plan completed, return current position (hold)
                joint_pos = curr_obs["joint_position"]
                gripper_val = curr_obs["gripper_position"][0] if len(curr_obs["gripper_position"]) > 0 else self._last_gripper_state
                action = np.concatenate([joint_pos.flatten(), np.array([gripper_val])])
                return self._make_result(action, curr_obs)

            # Get next action from plan
            step = self._plan[self._current_plan_step]

            if step["type"] == "gripper":
                # Handle gripper action
                self._gripper_action_pending = step["action"]
                self._gripper_action_steps_remaining = self._gripper_action_steps

                joint_pos = curr_obs["joint_position"]
                gripper_val = 1.0 if self._gripper_action_pending == "close" else 0.0
                self._last_gripper_state = gripper_val
                action = np.concatenate([joint_pos.flatten(), np.array([gripper_val])])
                return self._make_result(action, curr_obs)
            else:
                # Handle trajectory action
                self._current_trajectory = step["positions"]
                self._current_waypoint_idx = 0
                self._current_plan_step += 1

        # Return next waypoint from current trajectory
        waypoint = self._current_trajectory[self._current_waypoint_idx]
        self._current_waypoint_idx += 1

        # Ensure waypoint has correct shape (7 joints + 1 gripper)
        if waypoint.shape[0] == 7:
            action = np.concatenate([waypoint, np.array([self._last_gripper_state])])
        else:
            action = waypoint

        return self._make_result(action, curr_obs)

    def _make_result(self, action: np.ndarray, curr_obs: dict) -> dict:
        """Create the result dictionary.

        Args:
            action: Action array (8,)
            curr_obs: Extracted observation

        Returns:
            Result dictionary with action and visualization
        """
        # Create visualization
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
        """Extract observation data from environment observation.

        Args:
            obs_dict: Raw observation dictionary from environment

        Returns:
            Extracted observation dictionary
        """
        policy = obs_dict["policy"]

        # Extract images
        right_image = policy["external_cam"][0].clone().detach().cpu().numpy()
        wrist_image = policy["wrist_cam"][0].clone().detach().cpu().numpy()

        # Extract proprioceptive state
        joint_position = policy["arm_joint_pos"].clone().detach().cpu().numpy()
        gripper_position = policy["gripper_pos"].clone().detach().cpu().numpy()

        return {
            "right_image": right_image,
            "wrist_image": wrist_image,
            "joint_position": joint_position,
            "gripper_position": gripper_position,
        }

    def close(self) -> None:
        """Close the websocket connection."""
        if self._ws is not None:
            self._ws.close()
            self._ws = None

