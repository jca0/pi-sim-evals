import torch
from typing import Optional, Dict, Any, TYPE_CHECKING
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject
import numpy as np
from .gemini_helpers import query_gemini, convert_np_to_bytes
import json

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.assets import RigidObject

def check_cube_in_bowl(
    env: "ManagerBasedRLEnv",
    cube_cfg: "SceneEntityCfg" = SceneEntityCfg("rubiks_cube"),
    bowl_cfg: "SceneEntityCfg" = SceneEntityCfg("_24_bowl"),
    max_x_threshold: float = 0.05,
    max_y_threshold: float = 0.05,
    max_z_threshold: float = 0.05,
) -> bool:
    """
    Check if a cube is in a bowl
    """

    from isaaclab.managers import SceneEntityCfg
    from isaaclab.assets import RigidObject

    cube: RigidObject = env.scene[cube_cfg.name]
    bowl: RigidObject = env.scene[bowl_cfg.name]
    
    cube_pos = cube.data.root_pos_w - env.scene.env_origins
    bowl_pos = bowl.data.root_pos_w - env.scene.env_origins
    
    x_diff = torch.abs(cube_pos[:, 0] - bowl_pos[:, 0])
    y_diff = torch.abs(cube_pos[:, 1] - bowl_pos[:, 1])
    z_diff = cube_pos[:, 2] - bowl_pos[:, 2]

    in_bowl = torch.logical_and(x_diff < max_x_threshold, y_diff < max_y_threshold)
    in_bowl = torch.logical_and(in_bowl, z_diff < max_z_threshold)
    in_bowl = torch.logical_and(in_bowl, z_diff > 0)

    return in_bowl

def gemini_check()


class TaskChecker:
    def __init__(self, scene: int, use_vlm: bool = False):
        self.scene = scene
        self.use_vlm = use_vlm

        self.instructions = {
            1: "put the cube in the bowl",
        }

        self.geometric_checkers = {
            1: check_cube_in_bowl,
        }

    def check(self, env: ManagerBasedRLEnv):
        checker_func = self.geometric_checkers[self.scene]
        result = checker_func(env)
        print("CHECKER RESULT: ", result)
        is_complete = bool(result)
        return is_complete

def get_checker(scene: int):
    return TaskChecker(scene)