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
import gymnasium as gym
import torch
import cv2
import mediapy
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from sim_evals.inference.droid_jointpos import Client as DroidJointPosClient

# Franka Panda home joint positions (7 joints) + gripper open (1)
HOME_JOINT_POS = np.array([0.0, -np.pi/5, 0.0, -4*np.pi/5, 0.0, 3*np.pi/5, 0.0, 0.0])
from dynamic_prompting.subtask_manager import SubtaskManager, decompose_task
from dynamic_prompting.progress_monitor import ProgressMonitor


def main(
        episodes:int = 1,
        headless: bool = True,
        scene: int = 1,
        length: int = 30,
        dynamic_prompting: bool = False,
        check_every_n_steps: int = 15,
        ):
    # launch omniverse app with arguments (inside function to prevent overriding tyro)
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = headless
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # All IsaacLab dependent modules should be imported after the app is launched
    import sim_evals.environments # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg


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
            instruction = "put the mug on top of the sugar box"
        case 5:
            instruction = "put the red cube in the bowl, put the blue cube in the bowl, take the red cube out of the bowl, then put the green cube in the bowl"
        case _:
            raise ValueError(f"Scene {scene} not supported")
        
    env_cfg.episode_length_s = length
    env_cfg.set_scene(scene)
    env = gym.make("DROID", cfg=env_cfg)

    obs, _ = env.reset()
    obs, _ = env.reset() # need second render cycle to get correctly loaded materials
    client = DroidJointPosClient()

    # Dynamic prompting setup
    manager = None
    monitor = None
    if dynamic_prompting:
        plan = decompose_task(instruction)
        print(f"Task decomposition: {[s.instruction for s in plan.subtasks]}")
        manager = SubtaskManager(plan)
        monitor = ProgressMonitor(
            check_every_n_steps=check_every_n_steps,
        )

    video_dir = Path("runs") / datetime.now().strftime("%Y-%m-%d") / datetime.now().strftime("%H-%M-%S")
    video_dir.mkdir(parents=True, exist_ok=True)
    video = []
    ep = 0
    max_steps = env.env.max_episode_length
    with torch.no_grad():
        for ep in range(episodes):
            for _ in tqdm(range(max_steps), desc=f"Episode {ep+1}/{episodes}"):
                # If all subtasks done, send home position instead of querying VLA
                if manager and manager.is_done():
                    action = torch.tensor(HOME_JOINT_POS, dtype=torch.float32)[None]
                    obs, _, term, trunc, _ = env.step(action)
                    if term or trunc:
                        break
                    continue

                # Use dynamic subtask instruction or the fixed one
                current_instruction = instruction
                if manager and not manager.is_done():
                    current_instruction = manager.current_instruction()

                ret = client.infer(obs, current_instruction)
                if not headless:
                    cv2.imshow("Right Camera", cv2.cvtColor(ret["viz"], cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
                video.append(ret["viz"])
                action = torch.tensor(ret["action"])[None]
                obs, _, term, trunc, _ = env.step(action)

                # Progress monitoring
                if manager and monitor and not manager.is_done():
                    frame = obs["policy"]["external_cam"][0].cpu().numpy()
                    monitor.set_frame(frame)
                    manager.step()

                    if monitor.should_check():
                        result = monitor.check_completion(current_instruction)
                        if result["completed"]:
                            print(f"Completed: {manager.status()} | {result['reason']}")
                            manager.advance()
                            if manager.is_done():
                                print("All subtasks completed")
                        elif manager.exceeded_subtask_limit():
                            print(f"Timed out: {manager.status()}")
                            manager.advance()

                if term or trunc:
                    break

            client.reset()
            if manager:
                manager.reset()
            if monitor:
                monitor.reset()
            mediapy.write_video(
                video_dir / f"episode_{ep}.mp4",
                video,
                fps=15,
            )
            video = []

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    args = tyro.cli(main)
