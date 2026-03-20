"""
Calibration script: generate prompt variations for a scene, roll each one out
with dynamic prompting, judge success, and save results.

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
from dynamic_prompting.subtask_manager import SubtaskManager, decompose_task
from dynamic_prompting.progress_monitor import ProgressMonitor
from dynamic_prompting.prompt_calibration import (
    generate_prompt_variations,
    judge_success,
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
    include_original: bool = True,
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

    # Generate prompt variations
    print(f"Generating {n_variations} prompt variations for: \"{base_instruction}\"")
    variations = generate_prompt_variations(base_instruction, n=n_variations)
    if include_original:
        variations = [base_instruction] + variations
    print(f"Testing {len(variations)} prompts:")
    for i, v in enumerate(variations):
        print(f"  [{i}] {v}")

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
        for var_idx, var_instruction in enumerate(variations):
            print(f"\n{'='*60}")
            print(f"[{var_idx+1}/{len(variations)}] Testing: \"{var_instruction}\"")
            print(f"{'='*60}")

            # Decompose this variation into subtasks
            plan = decompose_task(var_instruction)
            subtask_strs = [s.instruction for s in plan.subtasks]
            print(f"  Subtasks: {subtask_strs}")

            manager = SubtaskManager(plan)
            monitor = ProgressMonitor(check_every_n_steps=check_every_n_steps)
            video = []
            last_frame = None

            for step in tqdm(range(max_steps), desc=f"Variation {var_idx+1}"):
                current_instruction = var_instruction
                if not manager.is_done():
                    current_instruction = manager.current_instruction()

                ret = client.infer(obs, current_instruction)
                if not headless:
                    cv2.imshow("Right Camera", cv2.cvtColor(ret["viz"], cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
                video.append(ret["viz"])
                action = torch.tensor(ret["action"])[None]
                obs, _, term, trunc, _ = env.step(action)

                last_frame = obs["policy"]["external_cam"][0].cpu().numpy()

                if not manager.is_done():
                    monitor.set_frame(last_frame)
                    manager.step()

                    if monitor.should_check():
                        result = monitor.check_completion(current_instruction)
                        if result["completed"]:
                            print(f"  Subtask completed: {manager.status()} | {result['reason']}")
                            manager.advance()
                            if manager.is_done():
                                print("  All subtasks completed")
                        elif manager.exceeded_subtask_limit():
                            print(f"  Subtask timed out: {manager.status()}")
                            manager.advance()

                if term or trunc:
                    break

            # Judge overall success from final frame
            verdict = judge_success(last_frame, base_instruction)
            status = "SUCCESS" if verdict["success"] else "FAIL"
            print(f"  Result: [{status}] {verdict['reason']}")

            # Save video
            video_path = video_dir / f"var_{var_idx}_{status.lower()}.mp4"
            mediapy.write_video(video_path, video, fps=15)

            # Log result
            cal_log.save_result(
                scene=scene,
                instruction=var_instruction,
                subtasks=subtask_strs,
                success=verdict["success"],
                reason=verdict["reason"],
                video_path=str(video_path),
            )

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
