# src/inference/rmpflow_client.py
import numpy as np
from .abstract_client import InferenceClient

class RmpflowClient(InferenceClient):
    def __init__(self):
        # You will initialize the RMPflow controller *after* Isaac app + env are created,
        # because you need access to the robot articulation / stage.
        self.initialized = False
        self.robot = None
        self.controller = None
        self.default_gripper = 0.0  # open

    def init_with_env(self, env):
        """
        Call this once after env is created.
        `env.env` is the underlying IsaacLab env (ManagerBasedRLEnv).
        """
        if self.initialized:
            return

        # Import Isaac Sim RMPflow pieces
        from omni.isaac.motion_generation import RmpFlow, ArticulationMotionPolicy
        from omni.isaac.core import World

        # Get the underlying scene / world
        isaac_env = env.env
        # You may need to get the robot prim path from your USD, here it's /World/envs/env_0/robot
        robot_prim_path = isaac_env.scene["robot"].prim_path  # check this is correct

        # Create a World handle if needed and wrap robot art
        world = World.instance()
        robot = world.scene.get_object(robot_prim_path)

        # Load an RMPflow config for Franka (Lula config YAML)
        # You need to point this to the RMPflow config shipped with Isaac for Franka
        rmpflow_cfg_path = "/path/to/franka_rmpflow.yaml"

        rmpflow = RmpFlow(
            robot_description_path=rmpflow_cfg_path,
            urdf_path=None,  # when using USD-based description; otherwise set URDF
            end_effector_frame_name="panda_hand",  # or your EE frame
        )
        controller = ArticulationMotionPolicy(robot, rmpflow)

        self.robot = robot
        self.controller = controller
        self.initialized = True

    def reset(self):
        # RMPflow is reactive; nothing persistent to reset by default.
        pass

    def infer(self, obs, instruction: str):
        """
        Compute one action: 7 arm joint targets + 1 gripper value in [0,1].
        """
        if not self.initialized:
            raise RuntimeError("Call init_with_env(env) on RmpflowClient before using it")

        # Extract current robot state from obs (you already have these terms configured)
        joint_pos = obs["policy"]["arm_joint_pos"][0].cpu().numpy()   # (7,)
        grip_pos = obs["policy"]["gripper_pos"][0].cpu().item()       # scalar in [0,1]

        # TODO: choose a target EE pose based on the current scene / instruction.
        # For now, just a hard-coded pose above the table:
        target_pos = np.array([0.4, 0.0, 0.4])  # x,y,z in world
        target_rot = np.array([1.0, 0.0, 0.0, 0.0])  # w,x,y,z

        # RMPflow: compute next joint command
        target = {
            "target_position": target_pos,
            "target_orientation": target_rot,
        }
        # controller returns joint position/velocity command depending on config
        joint_cmd = self.controller.forward(target)  # shape (num_joints,)

        # Take first 7 Panda joints and append gripper
        arm_action = joint_cmd[:7]
        gripper_action = np.array([self.default_gripper], dtype=np.float32)

        action = np.concatenate([arm_action, gripper_action], axis=0)  # (8,)

        return {"action": action, "viz": None, "viz_unscaled": None}