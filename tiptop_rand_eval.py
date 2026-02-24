"""Evaluation script that runs tiptop on randomized scene variants.

For a given scene number, discovers all scene{N}_rand{i}.usd files and runs
one episode per variant. All videos are saved in the same output folder.

Usage:
    # First, start the tiptop websocket server:
    # (in tiptop-robot) pixi run python -m tiptop.websocket_server --port 8765

    # Then run this evaluation:
    uv run python tiptop_rand_eval.py --scene 2 --ws-host localhost --ws-port 8765
"""

import argparse
import os
import time
from datetime import datetime
import numpy as np
from pathlib import Path

import cv2
import gymnasium as gym
import mediapy
import torch
import tyro
from tqdm import tqdm

from src.inference.tiptop_websocket import TiptopWebsocketClient, PlanningError


DATA_PATH = Path(__file__).parent / "assets"


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


def _discover_rand_variants(scene: int) -> list[str]:
    """Find all scene{N}_rand{i}.usd files and return sorted scene names for set_scene."""
    # Check both top-level assets/ and assets/scene{N}/ subdirectory
    patterns = [
        DATA_PATH / f"scene{scene}_rand*.usd",
        DATA_PATH / f"scene{scene}" / f"scene{scene}_rand*.usd",
    ]
    found = set()
    for pattern in patterns:
        for path in pattern.parent.glob(pattern.name):
            # Extract the part after "scene" and before ".usd"
            # e.g. "scene2_rand3.usd" -> need set_scene to resolve to this path
            # If in subdir: scene_name = "2/scene2_rand3" -> DATA_PATH/scene2/scene2_rand3.usd
            # If top-level: scene_name = "2_rand3" -> DATA_PATH/scene2_rand3.usd
            if path.parent == DATA_PATH:
                # Top-level: scene{scene}_rand{i}.usd -> scene_name = "{scene}_rand{i}"
                scene_name = path.stem.removeprefix("scene")
            else:
                # Subdirectory: scene{N}/scene{N}_rand{i}.usd -> scene_name = "{N}/{stem}"
                scene_name = f"{scene}/{path.stem}"
            found.add((scene_name, path.stem))
    # Sort by the rand index
    return sorted(found, key=lambda x: x[1])


def main(
    headless: bool = True,
    scene: int = 2,
    ws_host: str = "localhost",
    ws_port: int = 8765,
):
    """Run tiptop evaluation on all randomized variants of a scene.

    Args:
        headless: Run without GUI
        scene: Scene number (1-6)
        ws_host: Tiptop websocket server host
        ws_port: Tiptop websocket server port
    """
    # Discover randomized variants before launching the sim
    variants = _discover_rand_variants(scene)
    if not variants:
        print(f"No randomized variants found for scene {scene}.")
        print(f"Expected files like: {DATA_PATH}/scene{scene}_rand*.usd")
        print(f"                 or: {DATA_PATH}/scene{scene}/scene{scene}_rand*.usd")
        return

    variant_names = [v[1] for v in variants]
    print(f"Found {len(variants)} randomized variants for scene {scene}: {variant_names}")

    # Launch omniverse app
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Tiptop randomized evaluation")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = headless
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import src.environments  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    # Get task instruction for scene
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
            instruction = "stack the cubes"
        case _:
            raise ValueError(f"Scene {scene} not supported")

    video_dir = Path(__file__).parent / "runs" / datetime.now().strftime("%Y-%m-%d") / f"tiptop_scene{scene}_rand"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_fps = 15

    env = None
    client = None

    with torch.no_grad():
        for scene_name, variant_label in variants:
            print(f"\n--- Running variant: {variant_label} ---")

            # Recreate env for each variant to load the correct USD
            if env is not None:
                env.close()

            env_cfg = parse_env_cfg(
                "DROID",
                device=args_cli.device,
                num_envs=1,
                use_fabric=True,
            )
            env_cfg.set_scene(scene_name)
            env_cfg.episode_length_s = 1.0
            env = gym.make("DROID", cfg=env_cfg)

            obs, _ = env.reset()
            obs, _ = env.reset()  # Need second render cycle to get correctly loaded materials

            if client is None:
                print(f"Connecting to tiptop server at ws://{ws_host}:{ws_port}...")
                client = TiptopWebsocketClient(host=ws_host, port=ws_port)

            max_steps = env.env.max_episode_length

            obs, _ = env.reset()
            frame_idx = 0

            # Settle phase: run sim for ~1 second so objects settle into place
            settle_steps = 15
            for _ in range(settle_steps):
                hold_action = torch.cat([
                    obs["policy"]["arm_joint_pos"],
                    obs["policy"]["gripper_pos"],
                ], dim=-1).unsqueeze(0)
                obs, _, _, _, _ = env.step(hold_action)
            env.env.episode_length_buf[:] = 0

            video = []
            plan_failed = False
            for i in tqdm(range(max_steps), desc=variant_label):
                try:
                    ret = client.infer(obs, instruction)
                except PlanningError as e:
                    print(f"Planning failed for {variant_label}: {e}. Skipping.")
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
                continue

            video_path = video_dir / f"tiptop_{variant_label}.mp4"
            mediapy.write_video(video_path, video, fps=video_fps)
            print(f"Saved video to {video_path}")

    if client is not None:
        client.close()
    if env is not None:
        env.close()
    simulation_app.close()

    print(f"\nAll videos saved to {video_dir}")


if __name__ == "__main__":
    tyro.cli(main)
