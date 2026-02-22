"""
Randomize positions of the meat can and mug in scene 2, save photos and USD files.

Usage:
    uv run python randomize_scene2.py --headless
"""

import argparse
import numpy as np
import torch
import cv2
import gymnasium as gym
from pathlib import Path
from datetime import datetime


DATA_PATH = Path(__file__).parent / "assets"


def save_randomized_usd(source_usd: Path, output_usd: Path, new_positions: dict):
    """Save a copy of the scene USD with updated object positions.

    Args:
        source_usd: Path to the original scene2.usd
        output_usd: Path to write the modified USD
        new_positions: Dict mapping prim name -> (x, y, z) tuple
    """
    import shutil
    from pxr import Usd, Gf

    # Copy the original binary USD first, then modify in place
    shutil.copy2(str(source_usd), str(output_usd))

    stage = Usd.Stage.Open(str(output_usd))
    scene_prim = stage.GetPrimAtPath("/World")

    for child in scene_prim.GetChildren():
        name = child.GetName()
        if name in new_positions:
            pos = new_positions[name]
            translate_attr = child.GetAttribute("xformOp:translate")
            translate_attr.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))

    stage.GetRootLayer().Save()


def main(headless: bool = True, num_scenes: int = 3, start_index: int = 1, seed: int = 42):
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Randomize scene 2 objects")
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
    env_cfg.set_scene(2)
    env_cfg.episode_length_s = 60.0
    env = gym.make("DROID", cfg=env_cfg)

    # Initial resets to load materials
    obs, _ = env.reset()
    obs, _ = env.reset()

    # Get the rigid objects
    can_obj = env.env.scene["_10_potted_meat_can"]
    mug_obj = env.env.scene["_25_mug"]
    env_origins = env.env.scene.env_origins

    # Read original positions (local, relative to env origin)
    can_pos_orig = (can_obj.data.root_pos_w - env_origins)[0].cpu().numpy()
    mug_pos_orig = (mug_obj.data.root_pos_w - env_origins)[0].cpu().numpy()
    can_quat_orig = can_obj.data.root_quat_w[0].cpu().numpy()
    mug_quat_orig = mug_obj.data.root_quat_w[0].cpu().numpy()

    print(f"Original can position (local): {can_pos_orig}")
    print(f"Original mug position (local): {mug_pos_orig}")
    print(f"Original can quat: {can_quat_orig}")
    print(f"Original mug quat: {mug_quat_orig}")

    # Table surface bounds for randomization (x, y range on the table)
    # Based on the original object positions, the table is roughly:
    #   x: ~0.35 to ~0.55 (depth from robot)
    #   y: ~-0.20 to ~0.20 (left-right)
    #   z: stays at table height
    table_x_min, table_x_max = 0.35, 0.55
    table_y_min, table_y_max = -0.20, 0.20
    table_z = can_pos_orig[2]  # keep objects at table height

    # Minimum distance between objects to avoid overlap
    min_dist = 0.12

    output_dir = Path("runs") / "randomized_scene2" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_usd = DATA_PATH / "scene2.usd"

    rng = np.random.default_rng(seed=seed)

    for scene_idx in range(start_index, start_index + num_scenes):
        # Generate random positions ensuring minimum distance
        while True:
            can_x = rng.uniform(table_x_min, table_x_max)
            can_y = rng.uniform(table_y_min, table_y_max)
            mug_x = rng.uniform(table_x_min, table_x_max)
            mug_y = rng.uniform(table_y_min, table_y_max)
            dist = np.sqrt((can_x - mug_x) ** 2 + (can_y - mug_y) ** 2)
            if dist >= min_dist:
                break

        can_pos_new = np.array([can_x, can_y, table_z], dtype=np.float32)
        mug_pos_new = np.array([mug_x, mug_y, table_z], dtype=np.float32)

        print(f"\nScene {scene_idx}:")
        print(f"  Can position: {can_pos_new}")
        print(f"  Mug position: {mug_pos_new}")

        # Save randomized USD file
        usd_path = DATA_PATH / f"scene2_rand{scene_idx}.usd"
        save_randomized_usd(source_usd, usd_path, {
            "_10_potted_meat_can": can_pos_new,
            "_25_mug": mug_pos_new,
        })
        print(f"  Saved USD: {usd_path}")

        # Reset environment first
        obs, _ = env.reset()

        # Write new positions (need world frame = local + env_origins)
        origin = env_origins[0].cpu().numpy()

        can_root_state = can_obj.data.default_root_state.clone()
        can_root_state[0, :3] = torch.tensor(can_pos_new + origin, dtype=torch.float32, device=can_obj.device)
        can_root_state[0, 3:7] = torch.tensor(can_quat_orig, dtype=torch.float32, device=can_obj.device)
        can_obj.write_root_state_to_sim(can_root_state)

        mug_root_state = mug_obj.data.default_root_state.clone()
        mug_root_state[0, :3] = torch.tensor(mug_pos_new + origin, dtype=torch.float32, device=mug_obj.device)
        mug_root_state[0, 3:7] = torch.tensor(mug_quat_orig, dtype=torch.float32, device=mug_obj.device)
        mug_obj.write_root_state_to_sim(mug_root_state)

        # Step simulation a few times to let physics settle and render
        for _ in range(30):
            env.env.sim.step(render=True)

        # Update scene to get fresh sensor data
        env.env.scene.update(dt=env.env.sim.get_physics_dt())

        # Capture external camera image
        external_cam = env.env.scene["external_cam"]
        rgb = external_cam.data.output["rgb"][0].cpu().numpy()
        # rgb is (H, W, 4) RGBA - convert to BGR for cv2
        bgr = cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2BGR)

        # Save photo
        photo_path = output_dir / f"scene2_random_{scene_idx + 1}.png"
        cv2.imwrite(str(photo_path), bgr)
        print(f"  Saved photo: {photo_path}")

    print(f"\nAll {num_scenes} randomized scenes saved.")
    print(f"  USD files: {DATA_PATH}/scene2_rand[{start_index}-{start_index + num_scenes - 1}].usd")
    print(f"  Photos: {output_dir}/")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
