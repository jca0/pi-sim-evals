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
from datetime import datetime
from pathlib import Path
from typing import Literal

import cv2
import gymnasium as gym
import mediapy
import torch
import tyro
from tqdm import tqdm

from src.inference.tiptop_websocket import TiptopWebsocketClient


def main(
    episodes: int = 1,
    headless: bool = True,
    scene: int = 1,
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
            instruction = "put the meat can on the sugar box"
        case 5:
            instruction = "rearrange the cubes so that they spell 'REX'"
        case 6:
            instruction = "stack all the cubes on top of each other"
        case _:
            raise ValueError(f"Scene {scene} not supported")

    env_cfg.set_scene(scene)
    env_cfg.episode_length_s = 30.0  # LENGTH OF EPISODE
    env = gym.make("DROID", cfg=env_cfg)

    obs, _ = env.reset()
    obs, _ = env.reset()  # Need second render cycle to get correctly loaded materials

    # Connect to tiptop websocket server
    print(f"Connecting to tiptop server at ws://{ws_host}:{ws_port}...")
    client = TiptopWebsocketClient(host=ws_host, port=ws_port)

    video_dir = Path("runs") / datetime.now().strftime("%Y-%m-%d") / datetime.now().strftime("%H-%M-%S")
    video_dir.mkdir(parents=True, exist_ok=True)
    video = []
    max_steps = env.env.max_episode_length

    with torch.no_grad():
        for ep in range(episodes):
            for i in tqdm(range(max_steps), desc=f"Episode {ep+1}/{episodes}"):
                ret = client.infer(obs, instruction)
                if not headless:
                    cv2.imshow("Camera View", cv2.cvtColor(ret["viz"], cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
                video.append(ret["viz"])
                action = torch.tensor(ret["action"])[None]
                obs, _, term, trunc, _ = env.step(action)

                if term or trunc:
                    break

            client.reset()
            mediapy.write_video(
                video_dir / f"tiptop_ws_scene{scene}_ep{ep}.mp4",
                video,
                fps=15,
            )
            video = []
            print(f"Saved video to {video_dir / f'tiptop_ws_scene{scene}_ep{ep}.mp4'}")

    client.close()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    tyro.cli(main)

