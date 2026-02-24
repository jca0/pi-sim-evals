"""
Randomize object positions for scenes 1, 2, and 3. Saves photos and USD files.

Scene 1: rubiks_cube + bowl    ("put the cube in the bowl")
Scene 2: potted_meat_can + mug ("put the can in the mug")
Scene 3: banana + KLT bin      ("put banana in the bin")

Usage:
    uv run python randomize_scenes.py --scene 1 --headless
    uv run python randomize_scenes.py --scene 2 --headless
    uv run python randomize_scenes.py --scene 3 --headless
"""

import argparse
import numpy as np
import torch
import cv2
import gymnasium as gym
from pathlib import Path
from datetime import datetime


DATA_PATH = Path(__file__).parent / "assets"

# Per-scene config: (object_a_name, object_b_name, min_distance)
SCENE_CONFIGS = {
    1: {
        "objects": ["rubiks_cube", "_24_bowl"],
        "min_dist": 0.12,
    },
    2: {
        "objects": ["_10_potted_meat_can", "_25_mug"],
        "min_dist": 0.12,
    },
    3: {
        "objects": ["_11_banana", "small_KLT_visual_collision"],
        "min_dist": 0.15,
    },
}


def save_randomized_usd(source_usd: Path, output_usd: Path, new_positions: dict):
    """Save a copy of the scene USD with updated object positions.

    Args:
        source_usd: Path to the original scene USD
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


def main(scene: int = 2, headless: bool = True, num_scenes: int = 9, start_index: int = 1, seed: int = 42):
    if scene not in SCENE_CONFIGS:
        raise ValueError(f"Scene {scene} not supported. Choose from {list(SCENE_CONFIGS.keys())}")

    cfg = SCENE_CONFIGS[scene]
    obj_a_name, obj_b_name = cfg["objects"]
    min_dist = cfg["min_dist"]

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=f"Randomize scene {scene} objects")
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
    env_cfg.episode_length_s = 60.0
    env = gym.make("DROID", cfg=env_cfg)

    # Initial resets to load materials
    obs, _ = env.reset()
    obs, _ = env.reset()

    # Get the rigid objects
    obj_a = env.env.scene[obj_a_name]
    obj_b = env.env.scene[obj_b_name]
    env_origins = env.env.scene.env_origins

    # Read original positions (local, relative to env origin)
    pos_a_orig = (obj_a.data.root_pos_w - env_origins)[0].cpu().numpy()
    pos_b_orig = (obj_b.data.root_pos_w - env_origins)[0].cpu().numpy()
    quat_a_orig = obj_a.data.root_quat_w[0].cpu().numpy()
    quat_b_orig = obj_b.data.root_quat_w[0].cpu().numpy()

    print(f"Original {obj_a_name} position (local): {pos_a_orig}")
    print(f"Original {obj_b_name} position (local): {pos_b_orig}")
    print(f"Original {obj_a_name} quat: {quat_a_orig}")
    print(f"Original {obj_b_name} quat: {quat_b_orig}")

    # Table surface bounds for randomization (x, y range on the table)
    #   x: ~0.35 to ~0.55 (depth from robot)
    #   y: ~-0.20 to ~0.20 (left-right)
    #   z: stays at original height per object
    table_x_min, table_x_max = 0.35, 0.55
    table_y_min, table_y_max = -0.20, 0.20
    z_a = pos_a_orig[2]
    z_b = pos_b_orig[2]

    output_dir = Path("runs") / f"randomized_scene{scene}" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_usd = DATA_PATH / f"scene{scene}.usd"
    usd_output_dir = DATA_PATH / f"scene{scene}"
    usd_output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed=seed)

    for scene_idx in range(start_index, start_index + num_scenes):
        # Generate random positions ensuring minimum distance
        while True:
            ax = rng.uniform(table_x_min, table_x_max)
            ay = rng.uniform(table_y_min, table_y_max)
            bx = rng.uniform(table_x_min, table_x_max)
            by = rng.uniform(table_y_min, table_y_max)
            dist = np.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
            if dist >= min_dist:
                break

        pos_a_new = np.array([ax, ay, z_a], dtype=np.float32)
        pos_b_new = np.array([bx, by, z_b], dtype=np.float32)

        print(f"\nScene {scene_idx}:")
        print(f"  {obj_a_name} position: {pos_a_new}")
        print(f"  {obj_b_name} position: {pos_b_new}")

        # Save randomized USD file
        usd_path = usd_output_dir / f"scene{scene}_rand{scene_idx}.usd"
        save_randomized_usd(source_usd, usd_path, {
            obj_a_name: pos_a_new,
            obj_b_name: pos_b_new,
        })
        print(f"  Saved USD: {usd_path}")

        # Reset environment first
        obs, _ = env.reset()

        # Write new positions (need world frame = local + env_origins)
        origin = env_origins[0].cpu().numpy()

        a_root_state = obj_a.data.default_root_state.clone()
        a_root_state[0, :3] = torch.tensor(pos_a_new + origin, dtype=torch.float32, device=obj_a.device)
        a_root_state[0, 3:7] = torch.tensor(quat_a_orig, dtype=torch.float32, device=obj_a.device)
        obj_a.write_root_state_to_sim(a_root_state)

        b_root_state = obj_b.data.default_root_state.clone()
        b_root_state[0, :3] = torch.tensor(pos_b_new + origin, dtype=torch.float32, device=obj_b.device)
        b_root_state[0, 3:7] = torch.tensor(quat_b_orig, dtype=torch.float32, device=obj_b.device)
        obj_b.write_root_state_to_sim(b_root_state)

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
        photo_path = output_dir / f"scene{scene}_random_{scene_idx}.png"
        cv2.imwrite(str(photo_path), bgr)
        print(f"  Saved photo: {photo_path}")

    print(f"\nAll {num_scenes} randomized scenes saved.")
    print(f"  USD files: {usd_output_dir}/scene{scene}_rand[{start_index}-{start_index + num_scenes - 1}].usd")
    print(f"  Photos: {output_dir}/")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
