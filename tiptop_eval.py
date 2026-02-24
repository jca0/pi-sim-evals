"""Evaluation script that uses the tiptop websocket server for planning.

This script connects to a running tiptop websocket server, sends initial
observations (RGB, depth, camera params, task instruction), receives a
trajectory plan, and executes it in the Isaac Sim environment.

Usage:
    # First, start the tiptop websocket server:
    # (in tiptop-robot) pixi run python -m tiptop.websocket_server --port 8765

    # Then run this evaluation:
    uv run python tiptop_ws_eval.py --scene 1 --ws-host localhost --ws-port 8765
"""

import argparse
import os
import time
from datetime import datetime
import numpy as np
from pathlib import Path
from typing import Literal

import cv2
import gymnasium as gym
import mediapy
import torch
import tyro
import h5py
from tqdm import tqdm

from src.inference.tiptop_websocket import TiptopWebsocketClient


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
    episodes: int = 1,
    headless: bool = True,
    scene: int = 1,
    scene_variant: str = "",
    ws_host: str = "localhost",
    ws_port: int = 8765,
):
    """Run evaluation using tiptop websocket server.

    Args:
        episodes: Number of episodes to run
        headless: Run without GUI
        scene: Scene number (1-6)
        ws_host: Tiptop websocket server host
        ws_port: Tiptop websocket server port
    """
    # Launch omniverse app with arguments (inside function to prevent overriding tyro)
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Tiptop websocket evaluation")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = headless
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # All IsaacLab dependent modules should be imported after the app is launched
    import src.environments  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    # Initialize the env
    env_cfg = parse_env_cfg(
        "DROID",
        device=args_cli.device,
        num_envs=1,
        use_fabric=True,
    )

    # Get task instruction for scene
    instruction = None
    match scene:
        case 1:
            instruction = "put the cube in the bowl"
        case 2:
            instruction = "put the can in the mug"
        case 3:
            instruction = "put banana in the bin"
        case 4: 
            # instruction = "pack the cans on top of the sugar box"
            instruction = "put the meat can on the sugar box"
        case 5:
            instruction = "put three cubes into the bowl"
        case 6:
            instruction = "stack the cubes"
        case _:
            raise ValueError(f"Scene {scene} not supported")

    env_cfg.set_scene(f"{scene}{scene_variant}")
    env_cfg.episode_length_s = 15.0  # LENGTH OF EPISODE
    env = gym.make("DROID", cfg=env_cfg)

    obs, _ = env.reset()
    obs, _ = env.reset()  # Need second render cycle to get correctly loaded materials
    wrist_cam = env.env.scene["wrist_cam"]
    intrinsic_matrix = wrist_cam.data.intrinsic_matrices[0].cpu().numpy()

    # Connect to tiptop websocket server
    print(f"Connecting to tiptop server at ws://{ws_host}:{ws_port}...")
    client = TiptopWebsocketClient(host=ws_host, port=ws_port)

    video_dir = Path("runs") / datetime.now().strftime("%Y-%m-%d") / datetime.now().strftime("%H-%M-%S")
    video_dir.mkdir(parents=True, exist_ok=True)
    video = []
    max_steps = env.env.max_episode_length
    video_fps = 15

    with torch.no_grad():
        for ep in range(episodes):
            obs, _ = env.reset()
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
            plan_failed = False
            for i in tqdm(range(max_steps), desc=f"Episode {ep+1}/{episodes}"):
                try:
                    ret = client.infer(obs, instruction)
                except Exception as e:
                    print(f"Inference failed for episode {ep+1}: {e}. Skipping.")
                    plan_failed = True
                    break

                viz = np.concatenate([ret["right_image"], ret["wrist_image"]], axis=1)
                viz = _add_top_padding(viz, pad_px=40)
                elapsed_ms = int(frame_idx * 1000 / video_fps)
                _overlay_timer_ms(viz, elapsed_ms)
                if not headless:
                    cv2.imshow("Camera View", cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)

                video.append(viz)
                frame_idx += 1

                action = torch.tensor(ret["action"])[None]
                obs, _, term, trunc, _ = env.step(action)
                if term or trunc:
                    break

            client.reset()
            if plan_failed:
                video = []
                continue
            mediapy.write_video(
                video_dir / f"tiptop_scene{scene}_ep{ep}.mp4",
                video,
                fps=video_fps,
            )
            video = []
            print(f"Saved video to {video_dir / f'tiptop_scene{scene}_ep{ep}.mp4'}")

    client.close()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    tyro.cli(main)
