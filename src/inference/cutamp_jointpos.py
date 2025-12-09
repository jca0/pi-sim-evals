import tyro
import numpy as np
import torch
from PIL import Image
from openpi_client import websocket_client_policy, image_tools
import os
import cv2
from .abstract_client import InferenceClient
from typing import Literal
import pickle
import sys
import os
import matplotlib.pyplot as plt

CUROBO_PATH = "/home/ubuntu/curobo/src"
if CUROBO_PATH not in sys.path:
    sys.path.insert(0, CUROBO_PATH)

from curobo.types.base import TensorDeviceType
from curobo.types.robot import JointState
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

class Client(InferenceClient):
    def __init__(self, 
                ) -> None:
        
        with open("jing_cutamp_plan_v2.pkl", "rb") as f:
            self.curobo_plan = pickle.load(f)
        
        self.current_plan_step = 0
        self.current_trajectory = None
        self.current_waypoint_idx = 0
        self.gripper_action_pending = None
        self.gripper_action_steps_remaining = 0
        self.last_gripper_state = 0.0
        self.actual_history = []

        self._init_curobo_fk()

    def _init_curobo_fk(self):
        robot_file = "franka.yml"
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        tensor_args = TensorDeviceType(device=torch.device(device))
        
        robot_cfg_path = join_path(get_robot_configs_path(), robot_file)
        robot_cfg = load_yaml(robot_cfg_path)["robot_cfg"]
        
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            None,
            tensor_args,
            use_cuda_graph=False,
        )
        self.motion_gen = MotionGen(motion_gen_config)
        self.tensor_args = tensor_args

    def compute_fk(self, joint_positions):
        if joint_positions.ndim == 1:
            joint_positions = joint_positions.unsqueeze(0) if isinstance(joint_positions, torch.Tensor) else joint_positions[None]
        
        if isinstance(joint_positions, np.ndarray):
            joint_positions = torch.from_numpy(joint_positions).to(self.tensor_args.device)
        
        js = JointState.from_position(joint_positions)
        state = self.motion_gen.compute_kinematics(js)
        ee_pose = state.ee_pose
        
        return {
            'position': ee_pose.position.cpu().numpy(),
            'quaternion': ee_pose.quaternion.cpu().numpy()  # [w, x, y, z]
        }

    def compute_pose_error(self, desired_pose, actual_pose):
        pos_error = np.linalg.norm(desired_pose['position'] - actual_pose['position'])
        
        q_des = desired_pose['quaternion']  # [w, x, y, z]
        q_act = actual_pose['quaternion']  # [w, x, y, z]
        
        q_act_inv = np.array([q_act[0], -q_act[1], -q_act[2], -q_act[3]])
        
        q_rel = np.array([
            q_des[0] * q_act_inv[0] - q_des[1] * q_act_inv[1] - q_des[2] * q_act_inv[2] - q_des[3] * q_act_inv[3],
            q_des[0] * q_act_inv[1] + q_des[1] * q_act_inv[0] + q_des[2] * q_act_inv[3] - q_des[3] * q_act_inv[2],
            q_des[0] * q_act_inv[2] - q_des[1] * q_act_inv[3] + q_des[2] * q_act_inv[0] + q_des[3] * q_act_inv[1],
            q_des[0] * q_act_inv[3] + q_des[1] * q_act_inv[2] - q_des[2] * q_act_inv[1] + q_des[3] * q_act_inv[0]
        ])
        
        q_rel = q_rel / np.linalg.norm(q_rel)
        rot_error = 2 * np.arccos(np.clip(np.abs(q_rel[0]), 0, 1))
        
        return {
            'translation_error': pos_error,
            'rotation_error': rot_error
        }

    def visualize(self, request: dict):
        """
        Return the camera views how the model sees it
        """
        curr_obs = self._extract_observation(request)
        base_img = image_tools.resize_with_pad(curr_obs["right_image"], 224, 224)
        wrist_img = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
        combined = np.concatenate([base_img, wrist_img], axis=1)
        return combined

    def reset(self):
        self.current_plan_step = 0
        self.current_trajectory = None
        self.current_waypoint_idx = 0
        self.gripper_action_pending = None
        self.gripper_action_steps_remaining = 0
        self.last_gripper_state = 0.0 

    def infer(self, obs: dict, instruction: str) -> dict:
        """
        Infer the next action from the curobo plan
        """
        curr_obs = self._extract_observation(obs)
        
        # Handle gripper actions first if pending
        if self.gripper_action_pending is not None:
            if self.gripper_action_steps_remaining > 0:
                self.gripper_action_steps_remaining -= 1
                # Return current joint position with gripper action
                joint_pos = curr_obs["joint_position"]
                gripper_val = 1.0 if self.gripper_action_pending == "close" else 0.0
                self.last_gripper_state = gripper_val  # Update tracked state
                action = np.concatenate([joint_pos, np.array([gripper_val])])
                
                img1 = image_tools.resize_with_pad(curr_obs["right_image"], 224, 224)
                img2 = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
                both = np.concatenate([img1, img2], axis=1)
                
                return {"action": action, "viz": both, "right_image": curr_obs["right_image"], "wrist_image": curr_obs["wrist_image"]}
            else:
                self.last_gripper_state = 1.0 if self.gripper_action_pending == "close" else 0.0
                self.gripper_action_pending = None
                self.current_plan_step += 1

        # RECORD ACTUAL JOINT POSITIONS
        if self.current_trajectory is not None:
            self.actual_history.append(curr_obs["joint_position"])

        # PLOT ACTUAL VS EXPECTED JOINT POSITIONS
        if self.current_trajectory is not None and self.current_waypoint_idx >= len(self.current_trajectory):
            actual_arr = np.array(self.actual_history)
            expected_arr = self.current_trajectory
            
            if len(actual_arr) > 1:
                fig, axs = plt.subplots(7, 1, figsize=(10, 15))
                steps = np.arange(len(expected_arr))
                
                for joint_idx in range(7):
                    axs[joint_idx].plot(steps, expected_arr[:, joint_idx], 'r--', label='Expected')
                    # Use actual_arr[1:] to match the steps resulting from the actions
                    axs[joint_idx].plot(steps, actual_arr[:, joint_idx], 'b-', label='Actual')
                    axs[joint_idx].set_title(f'Joint {joint_idx}')
                    axs[joint_idx].legend()
                
                plt.tight_layout()
                plt.savefig(f"output/traj_plot_step_{self.current_plan_step}.png")
                plt.close()

                desired_final_joints = expected_arr[-1]  # Last waypoint
                actual_final_joints = actual_arr[-1]     # Last actual position
                
                # Compute forward kinematics
                desired_ee_pose = self.compute_fk(desired_final_joints)
                actual_ee_pose = self.compute_fk(actual_final_joints)
                
                # Compute errors
                errors = self.compute_pose_error(
                    {'position': desired_ee_pose['position'][0], 'quaternion': desired_ee_pose['quaternion'][0]},
                    {'position': actual_ee_pose['position'][0], 'quaternion': actual_ee_pose['quaternion'][0]}
                )
                
                print(f"\ntrajectory {self.current_plan_step} end-effector errors:")
                print(f"translation error: {errors['translation_error']*1000:.2f} mm")
                print(f"rotation error: {np.degrees(errors['rotation_error']):.2f} degrees")
            
            self.actual_history = []
        # END PLOTTING CODE
        
        # Check if we need to load a new action from the plan
        if self.current_trajectory is None or self.current_waypoint_idx >= len(self.current_trajectory):
            # Check if we've completed all plan steps
            if self.current_plan_step >= len(self.curobo_plan):
                # Plan completed, return current position (hold)
                joint_pos = curr_obs["joint_position"]
                gripper_val = curr_obs["gripper_position"][0]
                action = np.concatenate([joint_pos, np.array([gripper_val])])
                
                img1 = image_tools.resize_with_pad(curr_obs["right_image"], 224, 224)
                img2 = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
                both = np.concatenate([img1, img2], axis=1)
                
                return {"action": action, "viz": both, "right_image": curr_obs["right_image"], "wrist_image": curr_obs["wrist_image"]}
            
            # Get next action from plan
            action_dict = self.curobo_plan[self.current_plan_step]
            
            if action_dict["type"] == "gripper":
                # Handle gripper action
                if action_dict["action"] == "open":
                    self.gripper_action_pending = "open"
                    # Approximate 0.25 seconds at 15 fps = ~4 steps
                    self.gripper_action_steps_remaining = 20
                elif action_dict["action"] == "close":
                    self.gripper_action_pending = "close"
                    # Approximate 0.4 seconds at 15 fps = ~6 steps
                    self.gripper_action_steps_remaining = 20
                else:
                    raise ValueError(f"Unknown gripper action: {action_dict['action']}")
                
                # Return action with gripper state
                joint_pos = curr_obs["joint_position"]
                gripper_val = 1.0 if self.gripper_action_pending == "close" else 0.0
                self.last_gripper_state = gripper_val 
                action = np.concatenate([joint_pos, np.array([gripper_val])])
                
                img1 = image_tools.resize_with_pad(curr_obs["right_image"], 224, 224)
                img2 = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
                both = np.concatenate([img1, img2], axis=1)
                
                return {"action": action, "viz": both, "right_image": curr_obs["right_image"], "wrist_image": curr_obs["wrist_image"]}
            else:
                # Handle trajectory action
                full_trajectory = action_dict["plan"].position.cpu().numpy()
                self.current_trajectory = full_trajectory
                self.current_waypoint_idx = 0
                self.current_plan_step += 1
        
        # Return next waypoint from current trajectory
        waypoint = self.current_trajectory[self.current_waypoint_idx]
        self.current_waypoint_idx += 1
        
        # Ensure waypoint has correct shape (7 joints + 1 gripper)
        if waypoint.shape[0] == 7:
            gripper_val = self.last_gripper_state 
            action = np.concatenate([waypoint, np.array([gripper_val])])
        else:
            action = waypoint
        
        img1 = image_tools.resize_with_pad(curr_obs["right_image"], 224, 224)
        img2 = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
        both = np.concatenate([img1, img2], axis=1)
        
        return {"action": action, "viz": both, "right_image": curr_obs["right_image"], "wrist_image": curr_obs["wrist_image"]}

    def _extract_observation(self, obs_dict, *, save_to_disk=False):
        # Assign images
        right_image = obs_dict["policy"]["external_cam"][0].clone().detach().cpu().numpy()
        wrist_image = obs_dict["policy"]["wrist_cam"][0].clone().detach().cpu().numpy()

        # Capture proprioceptive state
        robot_state = obs_dict["policy"]
        joint_position = robot_state["arm_joint_pos"].clone().detach().cpu().numpy()
        gripper_position = robot_state["gripper_pos"].clone().detach().cpu().numpy()

        if save_to_disk:
            combined_image = np.concatenate([right_image, wrist_image], axis=1)
            combined_image = Image.fromarray(combined_image)
            combined_image.save("robot_camera_views.png")

        return {
            "right_image": right_image,
            "wrist_image": wrist_image,
            "joint_position": joint_position,
            "gripper_position": gripper_position,
        }

if __name__ == "__main__":
    client = Client()