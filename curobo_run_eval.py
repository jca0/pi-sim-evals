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

from src.inference.motion_planner import CuroboClient

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
    env_cfg.episode_length_s = 10.0 
    
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
    
    video = []
    max_steps = env.env.max_episode_length
    
    with torch.no_grad():
        for ep in range(episodes):
            task_completed = False
            for i in tqdm(range(max_steps), desc=f"Episode {ep+1}/{episodes}"):
                ret = client.infer(obs, instruction)
                
                if not headless:
                    cv2.imshow("Right Camera", cv2.cvtColor(ret["viz"], cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
                
                video.append(ret["viz"])
                action = torch.tensor(ret["action"])[None]
                obs, _, term, trunc, _ = env.step(action)

                if i % 30 == 0 and not task_completed:
                    task_completed = task_checker.check(env.env, obs)
                    if task_completed:
                        break
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
