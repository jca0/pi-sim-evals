import logging
import numpy as np
from typing import List, Optional
import torch

_log = logging.getLogger(__name__)


class IsaacSimRobotClient:
    """Robot client adapter for Isaac Sim that implements TiPToP's RobotClient interface.
    
    This client wraps Isaac Sim environment state and provides the same interface
    as real robot clients (BambooFrankaClient, UR5Client) for compatibility with
    TiPToP's planning and execution code.
    
    Note: In Isaac Sim, actual action execution happens through the environment's
    step() method. This client primarily provides state queries and stores
    trajectories for step-by-step execution by the inference client.
    """
    
    def __init__(self, env=None, obs: Optional[dict] = None):
        """Initialize the Isaac Sim robot client.
        
        Args:
            env: Isaac Sim environment instance (optional, for future use)
            obs: Current observation dict from Isaac Sim (optional, can be updated later)
        """
        self.env = env
        self._current_obs = obs
        self._current_trajectory = None
        self._trajectory_index = 0
        self._trajectory_durations = None
        self._gripper_state = 0.0  # Track current gripper state (0=open, 1=close)

        if obs is not None:
            self._update_gripper_state_from_obs(obs)

    def update_observation(self, obs: dict):
        """Update the current observation from Isaac Sim.
        
        This should be called whenever a new observation is available.
        
        Args:
            obs: Observation dict from Isaac Sim with keys:
                - "policy": dict containing "arm_joint_pos" and "gripper_pos"
        """
        self._current_obs = obs
        self._update_gripper_state_from_obs(obs)

    def _update_gripper_state_from_obs(self, obs: dict) -> None:
        gripper_pos = obs.get("policy", {}).get("gripper_pos")
        if gripper_pos is None:
            return

        if isinstance(gripper_pos, torch.Tensor):
            gripper_pos = gripper_pos.clone().detach().cpu().numpy()

        gripper_pos = np.asarray(gripper_pos)
        if gripper_pos.ndim > 0:
            gripper_pos = gripper_pos[0]
        self._gripper_state = float(gripper_pos)

    def _coerce_joint_array(self, joint_array) -> np.ndarray:
        if isinstance(joint_array, torch.Tensor):
            joint_array = joint_array.clone().detach().cpu().numpy()
        return np.asarray(joint_array, dtype=np.float32)
    
    def get_joint_positions(self) -> List[float]:
        """Get current joint positions from Isaac Sim observation.
        
        Returns:
            List of 7 joint positions in radians
        """
        if self._current_obs is None:
            raise RuntimeError("No observation available. Call update_observation() first.")
        
        # Extract joint positions from observation
        # Format: obs["policy"]["arm_joint_pos"] is a torch tensor
        joint_pos = self._current_obs["policy"]["arm_joint_pos"]

        # Convert to numpy if needed, then to list
        joint_pos = self._coerce_joint_array(joint_pos)
        
        # Handle batch dimension if present (take first element)
        if joint_pos.ndim > 1:
            joint_pos = joint_pos[0]
        
        return joint_pos.tolist()
    
    def open_gripper(self, speed: float = 1.0, force: float = 0.1) -> dict:
        """Set gripper to open state.
        
        In Isaac Sim, gripper control is handled through the action space.
        This method just updates the internal gripper state.
        
        Args:
            speed: Gripper speed (ignored in simulation, kept for interface compatibility)
            force: Gripper force (ignored in simulation, kept for interface compatibility)
            
        Returns:
            dict with "success" key
        """
        self._gripper_state = 0.0
        _log.debug(f"Gripper set to open (speed={speed}, force={force})")
        return {"success": True}
    
    def close_gripper(self, speed: float = 1.0, force: float = 0.1) -> dict:
        """Set gripper to closed state.
        
        In Isaac Sim, gripper control is handled through the action space.
        This method just updates the internal gripper state.
        
        Args:
            speed: Gripper speed (ignored in simulation, kept for interface compatibility)
            force: Gripper force (ignored in simulation, kept for interface compatibility)
            
        Returns:
            dict with "success" key
        """
        self._gripper_state = 1.0
        _log.debug(f"Gripper set to close (speed={speed}, force={force})")
        return {"success": True}
    
    def get_gripper_state(self) -> float:
        """Get current gripper state.
        
        Returns:
            Gripper state: 0.0 for open, 1.0 for closed
        """
        return self._gripper_state
    
    def execute_joint_impedance_path(
        self, 
        joint_confs: np.ndarray,
        joint_vels: Optional[np.ndarray],
        durations: List[float],
    ) -> dict:
        """Store trajectory for step-by-step execution in Isaac Sim.
        
        In Isaac Sim, trajectories are executed step-by-step through the
        environment's step() method. This method stores the trajectory so
        it can be executed by the inference client.
        
        Args:
            joint_confs: numpy array of shape [N, 7], joint angles in radians
            joint_vels: numpy array of shape [N, 7], joint velocities in rad/s (not used in Isaac Sim)
            durations: list of float, timestep durations (not used in Isaac Sim)
            
        Returns:
            dict with "success" key
        """
        # Validate inputs
        joint_confs = self._coerce_joint_array(joint_confs)
        if len(joint_confs) == 0:
            return {"success": True}

        if joint_confs.shape[1] != 7:
            raise ValueError(f"Expected 7 joints, got {joint_confs.shape[1]}")

        if joint_vels is not None:
            joint_vels = self._coerce_joint_array(joint_vels)
            if joint_vels.shape != joint_confs.shape:
                raise ValueError(
                    f"Velocity shape {joint_vels.shape} doesn't match position shape {joint_confs.shape}"
                )

        # Store trajectory for step-by-step execution
        self._current_trajectory = joint_confs.copy()
        self._trajectory_index = 0
        self._trajectory_durations = list(durations)
        
        _log.info(f"Stored trajectory with {len(joint_confs)} waypoints for execution")
        return {"success": True}
    
    def get_next_waypoint(self) -> Optional[np.ndarray]:
        """Get the next waypoint from the stored trajectory.
        
        Returns:
            Next waypoint as numpy array of shape (7,), or None if trajectory is complete
        """
        if self._current_trajectory is None:
            return None
        
        if self._trajectory_index >= len(self._current_trajectory):
            return None
        
        waypoint = self._current_trajectory[self._trajectory_index]
        self._trajectory_index += 1
        return waypoint
    
    def has_trajectory(self) -> bool:
        """Check if there's an active trajectory with remaining waypoints.
        
        Returns:
            True if trajectory exists and has remaining waypoints
        """
        if self._current_trajectory is None:
            return False
        return self._trajectory_index < len(self._current_trajectory)
    
    def reset_trajectory(self):
        """Reset the current trajectory (e.g., when starting a new plan)."""
        self._current_trajectory = None
        self._trajectory_index = 0
        self._trajectory_durations = None
    
    def close(self):
        """Cleanup (no-op for Isaac Sim, kept for interface compatibility)."""
        self.reset_trajectory()
        self._current_obs = None
