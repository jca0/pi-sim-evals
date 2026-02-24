"""
Example script for running 10 rollouts of a DROID policy on the example environment.

Usage:

First, make sure you download the simulation assets and unpack them into the root directory of this package.

Then, in a separate terminal, launch the policy server on localhost:8000 
-- make sure to set XLA_PYTHON_CLIENT_MEM_FRACTION to avoid JAX hogging all the GPU memory.

For example, to launch a pi0-FAST-DROID policy (with joint position control), 
run the command below in a separate terminal from the openpi "karl/droid_policies" branch:

XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_droid_jointpos --policy.dir=s3://openpi-assets-simeval/pi0_fast_droid_jointpos

Finally, run the evaluation script:

python run_eval.py --episodes 10 --headless
"""

import tyro
import argparse
import time
import gymnasium as gym
import torch
import cv2
import mediapy
import h5py
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from typing import Literal

from src.inference.droid_jointpos import Client as DroidJointPosClient


def _add_top_padding(image, pad_px: int = 40):
    if pad_px <= 0:
        return image
    h, w = image.shape[:2]
    padded = np.zeros((h + pad_px, w, 3), dtype=image.dtype)
    padded[pad_px:, :, :] = image
    return padded


def _overlay_timer_ms(image, elapsed_ms: int) -> None:
    text = f"t={elapsed_ms} ms"
    org = (10, 28)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, org, font, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, org, font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)


def main(
        episodes:int = 10,
        headless: bool = True,
        scene: int = 1,
        policy: Literal["pi0.5", "pi0"] = "pi0.5",
        ):
    # launch omniverse app with arguments (inside function to prevent overriding tyro)
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = headless
    args_cli.policy = policy
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # All IsaacLab dependent modules should be imported after the app is launched
    import src.environments # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from src.inference.termination_checker import get_checker


    # Initialize the env
    env_cfg = parse_env_cfg(
        "DROID",
        device=args_cli.device,
        num_envs=1,
        use_fabric=True,
    )
    instruction = None
    match scene:
        case 1:
            instruction = "put the cube in the bowl"
        case 2:
            instruction = "put the can in the mug"
        case 3:
            instruction = "put banana in the bin"
        case 4: 
            instruction = "put the meat can on the sugar box"
        case 5:
            instruction = "put three cubes into the bowl"
        case 6:
            instruction = "stack the cubes on top of each other"
        case _:
            raise ValueError(f"Scene {scene} not supported")
        
    env_cfg.set_scene(scene)
    env_cfg.episode_length_s = 60.0 # LENGTH OF EPISODE
    env = gym.make("DROID", cfg=env_cfg)
    wrist_camera = env.env.scene["wrist_cam"]
    # intrinsics = wrist_camera.data.intrinsic_matrices[0].cpu().numpy()
    # depth = wrist_camera.data.output["distance_to_image_plane"][0]
    # rgb = wrist_camera.data.output["rgb"][0]
    # pos_w = wrist_camera.data.pos_w[0].cpu().numpy()
    # quat_w_ros = wrist_camera.data.quat_w_ros[0].cpu().numpy()


    # Create output directory
    video_dir = Path("runs") / datetime.now().strftime("%Y-%m-%d") / datetime.now().strftime("%H-%M-%S")
    video_dir.mkdir(parents=True, exist_ok=True)

    
    obs, _ = env.reset()
    obs, _ = env.reset() # need second render cycle to get correctly loaded materials
    client = DroidJointPosClient(policy=policy)
    task_checker = get_checker(scene, vlm=False)
    video = []
    right_video = []
    wrist_video = []
    ep = 0
    max_steps = env.env.max_episode_length
    video_fps = 15
    with torch.no_grad():
        for ep in range(episodes):
            obs, _ = env.reset()
            task_completed = False
            frame_idx = 0
            # Settle phase: run sim for ~1 second so objects settle into place
            settle_steps = 15  # 15 steps at 15 Hz = 1 second
            for _ in range(settle_steps):
                hold_action = torch.cat([
                    obs["policy"]["arm_joint_pos"],
                    obs["policy"]["gripper_pos"],
                ], dim=-1).unsqueeze(0)
                obs, _, _, _, _ = env.step(hold_action)
            env.env.episode_length_buf[:] = 0  # don't count settle steps toward episode length
            for i in tqdm(range(max_steps), desc=f"Episode {ep+1}/{episodes}"):
                ret = client.infer(obs, instruction)
                viz = np.concatenate([ret["right_image"], ret["wrist_image"]], axis=1)
                viz = _add_top_padding(viz, pad_px=40)
                elapsed_ms = int(frame_idx * 1000 / video_fps)
                _overlay_timer_ms(viz, elapsed_ms)
                if not headless:
                    cv2.imshow("Right Camera", cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
                # video.append(ret["viz"])
                video.append(viz)
                frame_idx += 1
                # right_video.append(ret["right_image"])
                # wrist_video.append(ret["wrist_image"])
                action = torch.tensor(ret["action"])[None]
                obs, _, term, trunc, _ = env.step(action)

                # if i % 30 == 0 and not task_completed:
                #     task_completed = task_checker.check(env.env, obs)
                #     if task_completed:
                #         print("TASK COMPLETED")
                #         term = True

                if term or trunc:
                    break

            client.reset()
            mediapy.write_video(
                video_dir / f"{policy}_scene{scene}_ep{ep}.mp4",
                video,
                fps=video_fps,
            )
            # # added right and wrist videos
            # mediapy.write_video(
            #     video_dir / f"{policy}_scene{scene}_ep{ep}_right.mp4",
            #     right_video,
            #     fps=15,
            # )
            # mediapy.write_video(
            #     video_dir / f"{policy}_scene{scene}_ep{ep}_wrist.mp4",
            #     wrist_video,
            #     fps=15,
            # )
            video = []
            # # reset right and wrist videos
            # right_video = []
            # wrist_video = []

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    args = tyro.cli(main)
