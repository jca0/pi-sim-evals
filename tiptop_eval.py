"""Evaluation script that uses the tiptop websocket server for planning.

This script connects to a running tiptop websocket server, sends initial
observations (RGB, depth, camera params, task instruction), receives a
trajectory plan, and executes it in the Isaac Sim environment.

Usage:
    # First, start the tiptop websocket server:
    # (in tiptop-robot) pixi run python -m tiptop.websocket_server --port 8765

    # Then run this evaluation:
    uv run python tiptop_ws_eval.py --scene 1 --ws-host localhost --ws-port 8765

    # To record data in raw DROID format for finetuning:
    uv run python tiptop_eval.py --scene 1 --record --record-dir ~/openpi/datasets/sim
"""

import argparse
import os
from datetime import datetime
import numpy as np
from pathlib import Path

import cv2
import gymnasium as gym
import mediapy
import torch
import tyro
from tqdm import tqdm

from src.inference.tiptop_websocket import TiptopWebsocketClient
from src.recording import save_episode_droid_format, update_annotations


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
    # scene: int = 1,
    scene: str = "1",
    ws_host: str = "localhost",
    ws_port: int = 8765,
    record: bool = False,
    record_dir: str = os.path.expanduser("~/pi-sim-evals/recordings"),
):
    """Run evaluation using tiptop websocket server.

    Args:
        episodes: Number of episodes to run
        headless: Run without GUI
        scene: Scene identifier, e.g. "1", "2", "2_rand1", "2_rand2"
        ws_host: Tiptop websocket server host
        ws_port: Tiptop websocket server port
        record: If True, save episode data in raw DROID format for finetuning
        record_dir: Directory to save recorded episodes (same structure as ~/openpi/datasets)
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

    # Get task instruction for scene (extract base scene number, e.g. "2_rand1" -> 2)
    # base_scene = scene  # uncomment when scene is int
    base_scene = int(scene.split("_")[0])  # uncomment when scene is str
    instruction = None
    match base_scene:
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
            raise ValueError(f"Scene {scene} (base {base_scene}) not supported")

    env_cfg.set_scene(scene)
    env_cfg.episode_length_s = 60.0  # LENGTH OF EPISODE
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
    video_fps = 15

    record_base = Path(record_dir)
    if record:
        record_base.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for ep in range(episodes):
            obs, _ = env.reset()
            frame_idx = 0

            # Per-episode recording buffers
            ep_ext_images = []
            ep_wrist_images = []
            ep_joint_positions = []
            ep_gripper_positions = []
            ep_actions = []

            # Settle phase: run sim for ~2 seconds so objects settle into place
            settle_steps = int(2.0 * video_fps)
            for _ in range(settle_steps):
                hold_action = torch.cat([
                    obs["policy"]["arm_joint_pos"],
                    obs["policy"]["gripper_pos"],
                ], dim=-1).unsqueeze(0)
                obs, _, _, _, _ = env.step(hold_action)
            env.env.episode_length_buf[:] = 0  # don't count settle steps toward episode length

            for i in tqdm(range(max_steps), desc=f"Episode {ep+1}/{episodes}"):
                # Record observation BEFORE taking the action
                if record:
                    ep_ext_images.append(
                        obs["policy"]["external_cam"][0].cpu().numpy().astype(np.uint8)
                    )
                    ep_wrist_images.append(
                        obs["policy"]["wrist_cam"][0].cpu().numpy().astype(np.uint8)
                    )
                    # arm_joint_pos obs function already selects env 0 internally,
                    # so shape is (7,) not (1, 7) — don't index [0] again.
                    ep_joint_positions.append(
                        obs["policy"]["arm_joint_pos"].cpu().numpy().astype(np.float64)
                    )
                    ep_gripper_positions.append(
                        float(obs["policy"]["gripper_pos"].cpu().numpy().item())
                    )

                ret = client.infer(obs, instruction)

                # If tiptop failed to find a plan, skip rest of episode
                if client._plan is not None and len(client._plan) == 0:
                    print(f"Plan failed, skipping episode {ep+1}")
                    break

                # Stop when the full trajectory has been executed
                if client.plan_done:
                    print(f"Plan fully executed at step {frame_idx}")
                    break

                # Record the action that will be executed
                if record:
                    ep_actions.append(np.array(ret["action"], dtype=np.float64))

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

            # Save visualization video
            if video:
                mediapy.write_video(
                    video_dir / f"tiptop_scene{scene}_ep{ep}.mp4",
                    video,
                    fps=video_fps,
                )
                print(f"Saved video to {video_dir / f'tiptop_scene{scene}_ep{ep}.mp4'}")
            else:
                print(f"No frames recorded for episode {ep+1}, skipping video save")
            video = []

            # Save episode in raw DROID format
            if record and len(ep_actions) > 0:
                ts = datetime.now()
                # UUID must not contain underscores — the conversion script
                # extracts it from "metadata_<uuid>.json" by splitting on "_".
                ep_uuid = f"SIM+sim+{ts.strftime('%Y-%m-%d-%Hh-%Mm-%Ss')}"
                episode_dir = record_base / ts.strftime("%Y-%m-%d_%H-%M-%S")

                save_episode_droid_format(
                    episode_dir=episode_dir,
                    episode_uuid=ep_uuid,
                    instruction=instruction,
                    ext_images=ep_ext_images,
                    wrist_images=ep_wrist_images,
                    joint_positions=ep_joint_positions,
                    gripper_positions=ep_gripper_positions,
                    actions=ep_actions,
                    fps=video_fps,
                )
                update_annotations(
                    record_base / "aggregated-annotations-030724.json",
                    ep_uuid,
                    instruction,
                )

    client.close()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    tyro.cli(main)
