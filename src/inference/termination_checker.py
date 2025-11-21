import torch
from typing import Optional, Dict, Any, TYPE_CHECKING
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject
import numpy as np
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
from src.inference.gemini_helpers import convert_np_to_bytes, parse_json

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_ID = "gemini-robotics-er-1.5-preview"

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.assets import RigidObject

def check_object_in_container(
    env: "ManagerBasedRLEnv",
    object_cfg: "SceneEntityCfg",
    container_cfg: "SceneEntityCfg",
    max_x_threshold: float = 0.01,
    max_y_threshold: float = 0.01,
    max_z_threshold: float = 0.01,) -> bool:
    """
    Check if an object is in a container
    """

    from isaaclab.managers import SceneEntityCfg
    from isaaclab.assets import RigidObject

    object: RigidObject = env.scene[object_cfg.name]
    container: RigidObject = env.scene[container_cfg.name]
    
    object_pos = object.data.root_pos_w - env.scene.env_origins
    container_pos = container.data.root_pos_w - env.scene.env_origins
    
    x_diff = torch.abs(object_pos[:, 0] - container_pos[:, 0])
    y_diff = torch.abs(object_pos[:, 1] - container_pos[:, 1])
    z_diff = object_pos[:, 2] - container_pos[:, 2] # positive z means object is in container

    in_container = torch.logical_and(x_diff < max_x_threshold, y_diff < max_y_threshold)
    in_container = torch.logical_and(in_container, z_diff < max_z_threshold)
    in_container = torch.logical_and(in_container, z_diff > 0)

    return in_container


class TaskChecker:
    def __init__(self, scene: int, vlm: bool = False):
        self.scene = scene
        self.vlm = vlm
        self.instructions = {
            1: "put the cube in the bowl",
            2: "put the can in the mug",
            3: "put banana in the bin",
            4: "put the meat can on the sugar box",
            5: "rearrange the cubes so that they spell 'REX'",
            6: "stack all the cubes on top of each other",
        }
        self.geometric_checkers = {
            1: {"object_cfg": SceneEntityCfg("rubiks_cube"), "container_cfg": SceneEntityCfg("_24_bowl")},
            2: {"object_cfg": SceneEntityCfg("_10_potted_meat_can"), "container_cfg": SceneEntityCfg("_25_mug")},
            3: {"object_cfg": SceneEntityCfg("_11_banana"), "container_cfg": SceneEntityCfg("small_KLT_visual_collision"), "max_z_threshold": 0.1},
        }

    def gemini_check(self, obs: dict):
        prompt = f"""
        You are a task completion checker for a robot.
        You are given a task and two images of the robot's view of the scene.
        You need to check if the task is complete.
        The task is: {self.instructions[self.scene]}
        Return a boolean value in the following json format: {{"is_complete": <boolean>}}.
        The boolean value should be True if the task is complete, False otherwise.
        """
        right_image = obs["policy"]["external_cam"][0].clone().detach().cpu().numpy()
        wrist_image = obs["policy"]["wrist_cam"][0].clone().detach().cpu().numpy()
        exterior_img = convert_np_to_bytes(right_image)
        wrist_img = convert_np_to_bytes(wrist_image)
        exterior_img_bytes = types.Part.from_bytes(
            data=exterior_img,
            mime_type='image/png',
        )
        wrist_img_bytes = types.Part.from_bytes(
            data=wrist_img,
            mime_type='image/png',
        )

        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[
                exterior_img_bytes,
                wrist_img_bytes,
                prompt,
            ],
            config=types.GenerateContentConfig(
                temperature=0.5,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            )
        )
        parsed = parse_json(response.text)
        return json.loads(parsed)["is_complete"]

    def check(self, env: ManagerBasedRLEnv, obs: dict):
        if self.vlm:
            gemini_result = self.gemini_check(obs)
            print("gemini_result: ", gemini_result)
            return gemini_result
        else:
            checker_func = check_object_in_container
            object_cfg = self.geometric_checkers[self.scene]["object_cfg"]
            container_cfg = self.geometric_checkers[self.scene]["container_cfg"]
            max_z_threshold = self.geometric_checkers[self.scene].get("max_z_threshold", 0.01)
            result = checker_func(env, object_cfg, container_cfg, max_z_threshold)
            return bool(result)

def get_checker(scene: int, vlm: bool = False):
    return TaskChecker(scene, vlm)