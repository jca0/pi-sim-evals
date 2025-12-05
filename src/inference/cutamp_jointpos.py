import tyro
import numpy as np
from PIL import Image
from openpi_client import websocket_client_policy, image_tools
import os
import cv2
from .abstract_client import InferenceClient
from typing import Literal
import pickle
import sys
import os

CUROBO_PATH = "/home/ubuntu/curobo/src"
if CUROBO_PATH not in sys.path:
    sys.path.insert(0, CUROBO_PATH)

class Client(InferenceClient):
    def __init__(self, 
                ) -> None:
        
        with open("jing_cutamp_plan.pkl", "rb") as f:
            self.curobo_plan = pickle.load(f)
        
        # Initialize plan execution state
        self.current_plan_step = 0
        self.current_trajectory = None
        self.current_waypoint_idx = 0
        self.gripper_action_pending = None
        self.gripper_action_steps_remaining = 0

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
        # Reset plan execution state
        self.current_plan_step = 0
        self.current_trajectory = None
        self.current_waypoint_idx = 0
        self.gripper_action_pending = None
        self.gripper_action_steps_remaining = 0

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
                action = np.concatenate([joint_pos, np.array([gripper_val])])
                
                img1 = image_tools.resize_with_pad(curr_obs["right_image"], 224, 224)
                img2 = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
                both = np.concatenate([img1, img2], axis=1)
                
                return {"action": action, "viz": both, "right_image": curr_obs["right_image"], "wrist_image": curr_obs["wrist_image"]}
            else:
                # Gripper action completed, move to next step
                self.gripper_action_pending = None
                self.current_plan_step += 1
        
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
                    self.gripper_action_steps_remaining = 4
                elif action_dict["action"] == "close":
                    self.gripper_action_pending = "close"
                    # Approximate 0.4 seconds at 15 fps = ~6 steps
                    self.gripper_action_steps_remaining = 6
                else:
                    raise ValueError(f"Unknown gripper action: {action_dict['action']}")
                
                # Return action with gripper state
                joint_pos = curr_obs["joint_position"]
                gripper_val = 1.0 if self.gripper_action_pending == "close" else 0.0
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
            # Add gripper position (keep current gripper state)
            gripper_val = curr_obs["gripper_position"][0]
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