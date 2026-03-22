"""
Calibration script: generate subtask decomposition variations for a scene,
roll each one out with dynamic prompting, and log which subtask prompts
succeed or fail.

Usage:
    python run_calibration.py --scene 1 --n-variations 5 --headless

Results are saved to calibration_logs/scene_<N>.json
"""

import tyro
import argparse
import gymnasium as gym
import torch
import cv2
import mediapy
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from sim_evals.inference.droid_jointpos import Client as DroidJointPosClient
from dynamic_prompting.subtask_manager import SubtaskManager
from dynamic_prompting.progress_monitor import ProgressMonitor
from dynamic_prompting.prompt_calibration import (
    generate_decomposition_variations,
    CalibrationLog,
)


# Map scene number to the base instruction
SCENE_INSTRUCTIONS = {
    1: "put the cube in the bowl",
    2: "put the can in the mug",
    3: "put banana in the bin",
    4: "put the mug on top of the sugar box",
    5: "put 3 cubes in the bowl, take out 2, then put all the cubes back in the bowl",
}


def main(
    scene: int = 1,
    n_variations: int = 5,
    headless: bool = True,
    length: int = 30,
    check_every_n_steps: int = 15,
):
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = headless
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import sim_evals.environments  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    if scene not in SCENE_INSTRUCTIONS:
        raise ValueError(f"Scene {scene} not supported")

    base_instruction = SCENE_INSTRUCTIONS[scene]

    # Generate decomposition variations
    print(f"Generating {n_variations} decomposition variations for: \"{base_instruction}\"")
    plans = generate_decomposition_variations(base_instruction, n=n_variations)
    print(f"Testing {len(plans)} decompositions:")
    for i, plan in enumerate(plans):
        print(f"  [{i}] {plan.subtasks}")

    # Setup env
    env_cfg = parse_env_cfg("DROID", device=args_cli.device, num_envs=1, use_fabric=True)
    env_cfg.episode_length_s = length
    env_cfg.set_scene(scene)
    env = gym.make("DROID", cfg=env_cfg)
    obs, _ = env.reset()
    obs, _ = env.reset()

    client = DroidJointPosClient()
    cal_log = CalibrationLog()
    video_dir = Path("runs") / "calibration" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    video_dir.mkdir(parents=True, exist_ok=True)

    max_steps = env.env.max_episode_length

    with torch.no_grad():
        for plan_idx, plan in enumerate(plans):
            print(f"\n{'='*60}")
            print(f"[{plan_idx+1}/{len(plans)}] Subtasks: {plan.subtasks}")
            print(f"{'='*60}")

            manager = SubtaskManager(plan)
            monitor = ProgressMonitor(check_every_n_steps=check_every_n_steps)
            video = []

            for step in tqdm(range(max_steps), desc=f"Decomposition {plan_idx+1}"):
                current_instruction = manager.current_instruction()

                ret = client.infer(obs, current_instruction)
                if not headless:
                    cv2.imshow("Right Camera", cv2.cvtColor(ret["viz"], cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
                video.append(ret["viz"])
                action = torch.tensor(ret["action"])[None]
                obs, _, term, trunc, _ = env.step(action)

                if not manager.is_done():
                    frame = obs["policy"]["external_cam"][0].cpu().numpy()
                    monitor.set_frame(frame)

                    if monitor.should_check():
                        result = monitor.check_completion(current_instruction)
                        if result["completed"]:
                            print(f"  Subtask completed: {manager.status()} | {result['reason']}")
                            cal_log.log_subtask_result(
                                scene=scene,
                                subtask_prompt=current_instruction,
                                completed=True,
                                reason=result["reason"],
                            )
                            manager.advance()
                            if manager.is_done():
                                print("  All subtasks completed")

                if term or trunc:
                    break

            # Log remaining incomplete subtasks as failed
            while not manager.is_done():
                cal_log.log_subtask_result(
                    scene=scene,
                    subtask_prompt=manager.current_instruction(),
                    completed=False,
                    reason="episode ended before completion",
                )
                manager.advance()

            # Save video
            video_path = video_dir / f"decomp_{plan_idx}.mp4"
            mediapy.write_video(video_path, video, fps=15)

            # Reset for next variation
            client.reset()
            obs, _ = env.reset()

    # Print summary
    print(f"\n{'='*60}")
    print("CALIBRATION SUMMARY")
    print(f"{'='*60}")
    print(cal_log.summary(scene))
    print(f"\nResults saved to: {cal_log._log_path(scene)}")
    print(f"Videos saved to: {video_dir}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    tyro.cli(main)
