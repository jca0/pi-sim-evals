"""Evaluation script that runs tiptop on dynamically randomized scenes.

Loads a base scene once, then randomizes object positions at runtime for each
of 10 episodes. All videos are saved in the same output folder.

Usage:
    # First, start the tiptop websocket server:
    # (in tiptop-robot) pixi run python -m tiptop.websocket_server --port 8765

    # Then run this evaluation:
    uv run python tiptop_rand_eval.py --scene 2 --ws-host localhost --ws-port 8765
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path

import cv2
import gymnasium as gym
import mediapy
import numpy as np
import torch
import tyro
from tqdm import tqdm

from src.inference.tiptop_websocket import TiptopWebsocketClient, PlanningError
from src.scene_randomization import SCENE_CONFIG, randomize_objects


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
    headless: bool = True,
    scene: int = 2,
    episodes: int = 10,
    seed: int = 42,
    skip: tuple[int, ...] = (),
    ws_host: str = "localhost",
    ws_port: int = 8765,
):
    """Run tiptop evaluation with dynamically randomized object positions.

    Args:
        headless: Run without GUI
        scene: Scene number (1-6)
        episodes: Number of randomized episodes to run
        seed: Random seed for reproducibility
        ws_host: Tiptop websocket server host
        ws_port: Tiptop websocket server port
    """
    if scene not in SCENE_CONFIG:
        raise ValueError(f"Scene {scene} not supported. Available: {list(SCENE_CONFIG.keys())}")

    cfg = SCENE_CONFIG[scene]
    instruction = cfg["instruction"]

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

    env_cfg = parse_env_cfg(
        "DROID",
        device=args_cli.device,
        num_envs=1,
        use_fabric=True,
    )
    env_cfg.set_scene(scene)
    env_cfg.episode_length_s = 15.0
    env = gym.make("DROID", cfg=env_cfg)

    obs, _ = env.reset()
    obs, _ = env.reset()  # Need second render cycle to get correctly loaded materials

    print(f"Connecting to tiptop server at ws://{ws_host}:{ws_port}...")
    client = TiptopWebsocketClient(host=ws_host, port=ws_port)

    video_dir = Path("runs") / datetime.now().strftime("%Y-%m-%d") / f"tiptop_scene{scene}_rand"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_fps = 15
    max_steps = env.env.max_episode_length

    rng = np.random.default_rng(seed=seed)

    log_path = video_dir / "results.csv"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["episode", "seed", "success", "planning_time_s", "video_file"])
    log_file.flush()

    with torch.no_grad():
        for ep in range(episodes):
            print(f"\n--- Episode {ep}/{episodes} (seed={seed}, rng_step={ep}) ---")

            obs, _ = env.reset()

            # Randomize object positions (always advance RNG to keep layouts deterministic)
            positions = randomize_objects(
                env, cfg["objects"], rng,
                cfg["table_x"], cfg["table_y"], cfg["min_dist"],
                object_bounds=cfg.get("object_bounds"),
            )
            if ep in skip:
                print(f"  Skipping episode {ep}")
                continue
            for name, (x, y) in zip(cfg["objects"], positions):
                print(f"  {name}: ({x:.3f}, {y:.3f})")

            # Settle phase: temporarily extend episode length so env.step() doesn't auto-reset
            orig_episode_length_s = env.env.cfg.episode_length_s
            env.env.cfg.episode_length_s = orig_episode_length_s + 1.0
            settle_steps = 15  # 15 steps at 15 Hz = 1 second
            for _ in range(settle_steps):
                hold_action = torch.cat([
                    obs["policy"]["arm_joint_pos"],
                    obs["policy"]["gripper_pos"],
                ], dim=-1).unsqueeze(0)
                obs, _, _, _, _ = env.step(hold_action)
            env.env.cfg.episode_length_s = orig_episode_length_s
            env.env.episode_length_buf[:] = 0

            # Save snapshot of the scene after settling
            snapshot = obs["policy"]["external_cam"][0].cpu().numpy()
            snapshot_path = video_dir / f"tiptop_scene{scene}_rand{ep}_init.png"
            cv2.imwrite(str(snapshot_path), cv2.cvtColor(snapshot, cv2.COLOR_RGB2BGR))

            video = []
            frame_idx = 0
            plan_failed = False
            for i in tqdm(range(max_steps), desc=f"Episode {ep}/{episodes}"):
                try:
                    ret = client.infer(obs, instruction)
                except PlanningError as e:
                    print(f"Planning failed for episode {ep}: {e}. Skipping.")
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

            planning_time = client.last_planning_time
            client.reset()

            if plan_failed:
                log_writer.writerow([ep, seed, False, f"{planning_time:.2f}" if planning_time else "", ""])
                log_file.flush()
                continue

            video_path = video_dir / f"tiptop_scene{scene}_rand{ep}.mp4"
            mediapy.write_video(video_path, video, fps=video_fps)
            print(f"Saved video to {video_path}")
            log_writer.writerow([ep, seed, True, f"{planning_time:.2f}" if planning_time else "", video_path.name])
            log_file.flush()

    log_file.close()
    client.close()
    env.close()
    simulation_app.close()

    print(f"\nAll videos saved to {video_dir}")
    print(f"Results log: {log_path}")


if __name__ == "__main__":
    tyro.cli(main)
