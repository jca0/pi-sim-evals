"""
Example script for running curobo motion planning on the DROID environment.
"""

import tyro
import argparse
import gymnasium as gym
import torch
import cv2
import mediapy
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from typing import Literal

# Curobo imports
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

# DROID imports
from openpi_client import image_tools

print("DONE IMPORTING")

class CuroboClient:
    def __init__(self, env, device="cuda:0"):
        self.env = env
        self.device = device
        
        # Initialize Curobo
        self._init_curobo()
        
        # State machine state
        self.state = "INIT"
        self.plan = None
        self.plan_idx = 0
        self.gripper_state = 0.0 # 0.0 = Open, 1.0 = Closed
        self.gripper_timer = 0
        
        # Object names to look for in the scene
        self.cube_name = None
        self.bowl_name = None
        
        # Find objects in the Isaac Lab scene
        if hasattr(self.env.env, "scene"):
            scene_entities = self.env.env.scene.keys()
            for name in scene_entities:
                if "cube" in name.lower():
                    self.cube_name = name
                if "bowl" in name.lower() or "mug" in name.lower():
                    self.bowl_name = name
        
        print(f"Found cube: {self.cube_name}, bowl: {self.bowl_name}")

    def _init_curobo(self):
        # Configure MotionGen using standard Franka config
        robot_file = "franka.yml"
        world_file = "collision_table.yml" 
        
        tensor_args = TensorDeviceType(device=torch.device(self.device))
        
        robot_cfg_path = join_path(get_robot_configs_path(), robot_file)
        robot_cfg = load_yaml(robot_cfg_path)["robot_cfg"]
        
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            world_file,
            tensor_args,
            use_cuda_graph=True,
        )
        self.motion_gen = MotionGen(motion_gen_config)
        self.motion_gen.warmup()
        print("Curobo initialized and warmed up.")

    def _get_object_pose(self, name):
        if name is None:
            return None
        # Get ground truth pose from Isaac Lab scene
        # env.env.scene[name].data.root_pos_w is (num_envs, 3)
        pos = self.env.env.scene[name].data.root_pos_w[0]
        quat = self.env.env.scene[name].data.root_quat_w[0] # (w, x, y, z)
        
        # Convert to Curobo Pose [x, y, z, w, x, y, z]
        return Pose(
            position=pos.unsqueeze(0), 
            quaternion=quat.unsqueeze(0)
        )

    def plan_to_pose(self, current_q, target_pose):
        # The Curobo robot model seems to be configured for 7 DOF (arm only)
        # So we pass only the 7 arm joints.
        # current_q should be (7,) or (1,7)
        if current_q.dim() == 1:
            start_state = current_q.unsqueeze(0) # Shape (1, 7)
        else:
            start_state = current_q # Shape (1, 7)
            
        js = JointState.from_position(start_state)
        
        # Plan trajectory
        result = self.motion_gen.plan_single(js, target_pose.clone())
        
        if result.success.item():
            return result.interpolated_plan
        else:
            print("Motion planning failed!")
            return None

    def reset(self):
        self.state = "INIT"
        self.plan = None
        self.plan_idx = 0
        self.gripper_state = 0.0
        self.gripper_timer = 0

    def visualize(self, obs):
        right_image = obs["policy"]["external_cam"][0].cpu().numpy()
        wrist_image = obs["policy"]["wrist_cam"][0].cpu().numpy()
        
        img1 = image_tools.resize_with_pad(right_image, 224, 224)
        img2 = image_tools.resize_with_pad(wrist_image, 224, 224)
        return np.concatenate([img1, img2], axis=1)

    def infer(self, obs, instruction):
        # Extract observations
        # Use .view(-1) to flatten to 1D (7,) regardless of input shape (1,7) or (7,)
        joint_pos = obs["policy"]["arm_joint_pos"].to(self.device).view(-1)
        action_q = joint_pos.clone()
        
        # Helper to construct full action
        def get_action():
            # action: [q1...q7, gripper] (gripper > 0.5 is closed)
            act = torch.cat([action_q, torch.tensor([self.gripper_state], device=self.device)])
            return act.cpu().numpy()

        # --- State Machine ---
        if self.state == "INIT":
            if self.cube_name:
                self.state = "PLAN_PICK"
            else:
                print("Cube not found, staying idle.")
        
        elif self.state == "PLAN_PICK":
            cube_pose = self._get_object_pose(self.cube_name)
            target = cube_pose.clone()
            target.position[0, 2] += 0.3 # 30cm above cube (avoid table collision with long fingers)
            # Rotate gripper to point down (Quaternion: w, x, y, z)
            target.quaternion[0] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device)
            
            self.plan = self.plan_to_pose(joint_pos, target)
            if self.plan is not None:
                self.plan_idx = 0
                self.state = "EXECUTE_PRE_PICK"

        elif self.state == "EXECUTE_PRE_PICK":
            if self.plan is not None and self.plan_idx < self.plan.position.shape[0]:
                action_q = self.plan.position[self.plan_idx, :7]
                self.plan_idx += 1
            else:
                self.state = "PLAN_GRASP"
                
        elif self.state == "PLAN_GRASP":
            cube_pose = self._get_object_pose(self.cube_name)
            target = cube_pose.clone()
            target.position[0, 2] += 0.13 # At cube height + finger length offset (approx 11cm) + 2cm margin
            target.quaternion[0] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device)
            
            self.plan = self.plan_to_pose(joint_pos, target)
            if self.plan is not None:
                self.plan_idx = 0
                self.state = "EXECUTE_GRASP"

        elif self.state == "EXECUTE_GRASP":
            if self.plan is not None and self.plan_idx < self.plan.position.shape[0]:
                action_q = self.plan.position[self.plan_idx, :7]
                self.plan_idx += 1
            else:
                self.state = "CLOSE_GRIPPER"
                self.gripper_timer = 0

        elif self.state == "CLOSE_GRIPPER":
            self.gripper_state = 1.0 # Close
            self.gripper_timer += 1
            if self.gripper_timer > 20: # Wait for gripper to close
                self.state = "PLAN_LIFT"
        
        elif self.state == "PLAN_LIFT":
            cube_pose = self._get_object_pose(self.cube_name)
            target = cube_pose.clone()
            target.position[0, 2] += 0.3 # Lift up
            target.quaternion[0] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device)
            
            self.plan = self.plan_to_pose(joint_pos, target)
            if self.plan is not None:
                self.plan_idx = 0
                self.state = "EXECUTE_LIFT"

        elif self.state == "EXECUTE_LIFT":
            if self.plan is not None and self.plan_idx < self.plan.position.shape[0]:
                action_q = self.plan.position[self.plan_idx, :7]
                self.plan_idx += 1
            else:
                self.state = "PLAN_PLACE"

        elif self.state == "PLAN_PLACE":
            if self.bowl_name:
                bowl_pose = self._get_object_pose(self.bowl_name)
                target = bowl_pose.clone()
                target.position[0, 2] += 0.4 # Above bowl
                target.quaternion[0] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device)
                
                self.plan = self.plan_to_pose(joint_pos, target)
                if self.plan is not None:
                    self.plan_idx = 0
                    self.state = "EXECUTE_PLACE"
            else:
                self.state = "DONE"

        elif self.state == "EXECUTE_PLACE":
            if self.plan is not None and self.plan_idx < self.plan.position.shape[0]:
                action_q = self.plan.position[self.plan_idx, :7]
                self.plan_idx += 1
            else:
                self.state = "OPEN_GRIPPER"
                self.gripper_timer = 0

        elif self.state == "OPEN_GRIPPER":
            self.gripper_state = 0.0 # Open
            self.gripper_timer += 1
            if self.gripper_timer > 20:
                self.state = "DONE"

        return {
            "action": get_action(),
            "viz": self.visualize(obs)
        }

def main(
        episodes:int = 1,
        headless: bool = True,
        scene: int = 1,
        ):
    # Launch app
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description="Curobo Motion Planning Evaluation")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = headless
    
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Imports after app launch
    import src.environments # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from src.inference.termination_checker import get_checker

    print("DONE IMPORTING ISAAC")

    # Initialize env
    env_cfg = parse_env_cfg(
        "DROID",
        device=args_cli.device,
        num_envs=1,
        use_fabric=True,
    )
    
    instruction = "put the cube in the bowl"
    env_cfg.set_scene(scene)
    env_cfg.episode_length_s = 5.0 
    
    env = gym.make("DROID", cfg=env_cfg)
    
    obs, _ = env.reset()
    obs, _ = env.reset()
    
    # Initialize Curobo Client
    client = CuroboClient(env, device=args_cli.device)
    
    task_checker = get_checker(scene, vlm=False)

    # Use absolute path for video directory to avoid ffmpeg issues
    video_dir = Path("runs") / datetime.now().strftime("%Y-%m-%d") / datetime.now().strftime("%H-%M-%S")
    video_dir = video_dir.resolve()
    video_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving videos to: {video_dir}")
    
    video = []
    max_steps = env.env.max_episode_length

    print("STARTING EVALUATION")
    
    with torch.no_grad():
        for ep in range(episodes):
            task_completed = False
            for i in tqdm(range(max_steps), desc=f"Episode {ep+1}/{episodes}"):
                ret = client.infer(obs, instruction)
                # print(f"Client state: {client.state}")
                
                if not headless:
                    cv2.imshow("Right Camera", cv2.cvtColor(ret["viz"], cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
                
                video.append(ret["viz"])
                action = torch.tensor(ret["action"])[None]
                obs, _, term, trunc, _ = env.step(action)

                if i % 30 == 0 and not task_completed:
                    task_completed = task_checker.check(env.env, obs)
                    if task_completed:
                        print("TASK COMPLETED")

                if term or trunc:
                    break

            client.reset()
            # Convert path to string for mediapy
            video_path = str(video_dir / f"curobo_scene{scene}_ep{ep}.mp4")
            mediapy.write_video(
                video_path,
                video,
                fps=15,
            )
            video = []
            obs, _ = env.reset()

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    tyro.cli(main)
