import tyro
import numpy as np
from PIL import Image
from openpi_client import websocket_client_policy, image_tools
import os
import cv2
from .gemini_helpers import query_gemini, scale_bounding_boxes, plot_bounding_boxes, convert_np_to_bytes, scale_points, plot_points
from .abstract_client import InferenceClient
from typing import Literal

class Client(InferenceClient):
    def __init__(self, 
                remote_host:str = "localhost", 
                remote_port:int = 8000,
                open_loop_horizon:int = 8,
                policy: Literal["pi0.5", "pi0"] = "pi0.5" ) -> None:
        self.policy = policy
        if self.policy == "pi0.5":
            self.open_loop_horizon = 15
        elif self.policy == "pi0":
            self.open_loop_horizon = 8

        self.client = websocket_client_policy.WebsocketClientPolicy(
            remote_host, remote_port
        )
        
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        self.inference_count = 0

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
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None

    def infer(self, obs: dict, instruction: str) -> dict:
        """
        Infer the next action from the policy in a server-client setup
        """
        curr_obs = self._extract_observation(obs)
        if (
            self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.open_loop_horizon
        ):
            self.actions_from_chunk_completed = 0

            exterior_annotated_resized = image_tools.resize_with_pad(curr_obs["right_image"], 224, 224)
            wrist_annotated_resized = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
            
            request_data = {
                "observation/exterior_image_1_left": exterior_annotated_resized,
                "observation/wrist_image_left": wrist_annotated_resized,
                "observation/joint_position": curr_obs["joint_position"],
                "observation/gripper_position": curr_obs["gripper_position"],
                "prompt": instruction,
            }
            pred_chunk = self.client.infer(request_data)["actions"] # velocities

            if self.policy == "pi0.5":
                # convert velocities to joint positions
                dt = 1.0/15.0
                pred_chunk = pred_chunk.copy()
                pred_chunk[:, :-1] *= dt
                pred_chunk[:, :-1] = np.cumsum(pred_chunk[:, :-1], axis=0)
                pred_chunk[:, :-1] += curr_obs["joint_position"]
            
            self.pred_action_chunk = pred_chunk

            self.inference_count += 1

        action = self.pred_action_chunk[self.actions_from_chunk_completed]
        self.actions_from_chunk_completed += 1

        # binarize gripper action
        if action[-1].item() > 0.5:
            action = np.concatenate([action[:-1], np.ones((1,))])
        else:
            action = np.concatenate([action[:-1], np.zeros((1,))])

        img1 = image_tools.resize_with_pad(curr_obs["right_image"], 672, 672)
        img2 = image_tools.resize_with_pad(curr_obs["wrist_image"], 672, 672)
        both = np.concatenate([img1, img2], axis=1)

        # added right image and wrist image to dict
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
    import torch
    args = tyro.cli(Args)
    client = Client(args)
    fake_obs = {
        "splat": {
            "right_cam": np.zeros((224, 224, 3), dtype=np.uint8),
            "wrist_cam": np.zeros((224, 224, 3), dtype=np.uint8),
        },
        "policy": {
            "arm_joint_pos": torch.zeros((7,), dtype=torch.float32),
            "gripper_pos": torch.zeros((1,), dtype=torch.float32),

        },
    }
    fake_instruction = "pick up the object"

    import time

    start = time.time()
    client.infer(fake_obs, fake_instruction) # warm up
    num = 20
    for i in range(num):
        ret = client.infer(fake_obs, fake_instruction)
        print(ret["action"].shape)
    end = time.time()

    print(f"Average inference time: {(end - start) / num}")
