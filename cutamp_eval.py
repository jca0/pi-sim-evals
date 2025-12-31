"""
TiPToP + Simulation Integration

Run TiPToP planning on simulation observations and execute trajectories.

Usage:
1. Start the M2T2 and FoundationStereo servers (or use simulation depth)
2. python run_tiptop_sim.py --scene 1 --headless
"""

import tyro
import argparse
import gymnasium as gym
import torch
import cv2
import mediapy
import pickle
import asyncio
import aiohttp
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation

from src.inference.cutamp_jointpos import Client as CutampJointPosClient


def quat_to_rotation_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert quaternion [w,x,y,z] to 3x3 rotation matrix."""
    # scipy uses [x,y,z,w] format
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    return Rotation.from_quat(quat_xyzw).as_matrix()


def build_world_from_cam(pos_w: np.ndarray, quat_w_ros: np.ndarray) -> np.ndarray:
    """Build 4x4 world_from_cam transform from position and quaternion."""
    world_from_cam = np.eye(4)
    world_from_cam[:3, :3] = quat_to_rotation_matrix(quat_w_ros)
    world_from_cam[:3, 3] = pos_w
    return world_from_cam


def depth_to_xyz(
    depth: np.ndarray,
    fx: float, fy: float,
    cx: float, cy: float,
) -> np.ndarray:
    """Convert depth map to XYZ point cloud using camera intrinsics."""
    h, w = depth.shape
    u = np.arange(w, dtype=np.float32)
    v = np.arange(h, dtype=np.float32)
    u_grid, v_grid = np.meshgrid(u, v)
    
    z = depth
    x = (u_grid - cx) * z / fx
    y = (v_grid - cy) * z / fy
    
    return np.stack([x, y, z], axis=-1)


async def run_tiptop_perception_sim(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    world_from_cam: np.ndarray,
    task_instruction: str,
    joint_positions: np.ndarray,
):
    """
    Run TiPToP perception pipeline using simulation observations.
    
    Args:
        rgb: RGB image (H, W, 3) uint8
        depth: Depth map in meters (H, W)
        intrinsics: 3x3 camera intrinsic matrix
        world_from_cam: 4x4 camera-to-world transform
        task_instruction: Natural language task instruction
        joint_positions: Current 7-DOF joint positions
    """
    # Import TiPToP modules (assumes tiptop is installed)
    from tiptop.perception_wrapper import detect_and_segment
    from tiptop.perception.m2t2 import generate_grasps_async
    from tiptop.perception.utils import depth_to_xyz, get_o3d_pcd
    from tiptop.tiptop_run import process_scene_geometry, create_tamp_environment
    from tiptop.config import tiptop_cfg
    
    cfg = tiptop_cfg()
    VOXEL_DOWNSAMPLE_SIZE = 0.0075
    
    # Convert depth to point cloud in camera frame
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    xyz_map = depth_to_xyz(depth, fx, fy, cx, cy)
    
    # Transform to world frame
    xyz_map = xyz_map @ world_from_cam[:3, :3].T + world_from_cam[:3, 3]
    rgb_map = rgb.astype(np.float32) / 255.0
    
    # Create downsampled point cloud
    pcd = get_o3d_pcd(xyz_map, rgb_map, VOXEL_DOWNSAMPLE_SIZE)
    xyz_downsampled = np.asarray(pcd.points)
    rgb_downsampled = np.asarray(pcd.colors)
    
    # Run detection/segmentation and grasp generation in parallel
    async with aiohttp.ClientSession() as session:
        detection_task = detect_and_segment(rgb, task_instruction)
        grasp_task = generate_grasps_async(
            session, 
            cfg.perception.m2t2.url, 
            scene_xyz=xyz_downsampled, 
            scene_rgb=rgb_downsampled
        )
        detection_results, grasps = await asyncio.gather(detection_task, grasp_task)
    
    # Process scene geometry
    processed_scene = process_scene_geometry(
        xyz_map=xyz_map,
        rgb_map=rgb_map,
        masks=detection_results["masks"],
        bboxes=detection_results["bboxes"],
        grasps=grasps,
    )
    
    # Create TAMP environment
    env, all_surfaces = create_tamp_environment(
        processed_scene.object_meshes,
        processed_scene.table_cuboid,
        detection_results["grounded_atoms"],
    )
    
    return env, all_surfaces, processed_scene.grasps


def run_tiptop_planning(
    env,
    all_surfaces,
    grasps,
    q_init: np.ndarray,
    save_dir: Path,
):
    """
    Run cuTAMP planning given a TAMP environment.
    
    Returns the cutamp_plan ready for execution.
    """
    from tiptop.config import tiptop_cfg
    from tiptop.motion_planning import get_ik_solver, get_motion_gen
    from tiptop.workspace import workspace_cuboids
    from cutamp.algorithm import run_cutamp
    from cutamp.config import TAMPConfiguration
    from cutamp.constraint_checker import ConstraintChecker
    from cutamp.cost_reduction import CostReducer
    from cutamp.scripts.utils import default_constraint_to_mult, default_constraint_to_tol
    from cutamp.task_planning.constraints import StablePlacement
    from curobo.geom.types import Cuboid, WorldConfig
    
    cfg = tiptop_cfg()
    
    # Setup cuRobo world config
    cuboids = [
        *workspace_cuboids(),
        Cuboid(name="table", dims=[0.01, 0.01, 0.01], pose=[99.99, 99.9, 99.9, 1.0, 0.0, 0.0, 0.0]),
    ]
    world_cfg = WorldConfig(cuboid=cuboids)
    
    # Initialize IK solver and motion generator
    num_particles = 256
    collision_activation_distance = 0.0
    ik_solver = get_ik_solver(world_cfg, num_particles)
    motion_gen = get_motion_gen(world_cfg, collision_activation_distance=collision_activation_distance)
    
    # Setup TAMP configuration
    config = TAMPConfiguration(
        num_particles=num_particles,
        max_loop_dur=30.0,
        num_opt_steps=500,
        m2t2_grasps=True,
        prop_satisfying_break=0.1,
        robot=cfg.robot.type,
        curobo_plan=True,
        warmup_ik=False,
        warmup_motion_gen=False,
    )
    
    # Setup constraint tolerances
    constraint_to_mult = default_constraint_to_mult.copy()
    constraint_to_tol = default_constraint_to_tol.copy()
    for surface in all_surfaces:
        constraint_to_tol[StablePlacement.type][f"{surface.name}_in_xy"] = 1e-2
        constraint_to_tol[StablePlacement.type][f"{surface.name}_support"] = 1e-2
        constraint_to_mult[StablePlacement.type][f"{surface.name}_support"] = 1.0
    
    cost_reducer = CostReducer(default_constraint_to_mult.copy())
    constraint_checker = ConstraintChecker(constraint_to_tol)
    
    # Run cuTAMP
    cutamp_plan, num_satisfying = run_cutamp(
        env,
        config,
        cost_reducer,
        constraint_checker,
        q_init=q_init,
        ik_solver=ik_solver,
        grasps=grasps,
        motion_gen=motion_gen,
        experiment_dir=save_dir / "cutamp",
    )
    
    return cutamp_plan


def extract_sim_observation(obs: dict, wrist_cam):
    """Extract observation data from simulation in TiPToP-compatible format."""
    # RGB from external camera (for perception)
    external_rgb = obs["policy"]["external_cam"][0].cpu().numpy()
    
    # Also get wrist camera for additional views
    wrist_rgb = obs["policy"]["wrist_cam"][0].cpu().numpy()
    
    # Depth from wrist camera (in meters)
    depth = wrist_cam.data.output["distance_to_image_plane"][0].cpu().numpy()
    
    # Camera intrinsics
    intrinsics = wrist_cam.data.intrinsic_matrices[0].cpu().numpy()
    
    # Camera extrinsics (world_from_cam)
    pos_w = wrist_cam.data.pos_w[0].cpu().numpy()
    quat_w_ros = wrist_cam.data.quat_w_ros[0].cpu().numpy()
    world_from_cam = build_world_from_cam(pos_w, quat_w_ros)
    
    # Joint positions
    joint_pos = obs["policy"]["arm_joint_pos"][0].cpu().numpy()
    
    return {
        "external_rgb": external_rgb,
        "wrist_rgb": wrist_rgb,
        "depth": depth,
        "intrinsics": intrinsics,
        "world_from_cam": world_from_cam,
        "joint_positions": joint_pos,
    }


def main(
    episodes: int = 1,
    headless: bool = True,
    scene: int = 1,
    use_sim_depth: bool = True,  # Use simulation depth instead of FoundationStereo
):
    """
    Run TiPToP planning on simulation initial observation, then execute.
    """
    # Launch Isaac Sim
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = headless
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Import Isaac Lab modules after app launch
    import src.environments  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from src.inference.termination_checker import get_checker

    # Get instruction for scene
    instructions = {
        1: "put the cube in the bowl",
        2: "put the can in the mug",
        3: "put banana in the bin",
        4: "put the meat can on the sugar box",
        5: "rearrange the cubes so that they spell 'REX'",
        6: "stack all the cubes on top of each other",
    }
    if scene not in instructions:
        raise ValueError(f"Scene {scene} not supported")
    instruction = instructions[scene]

    # Initialize environment
    env_cfg = parse_env_cfg("DROID", device=args_cli.device, num_envs=1, use_fabric=True)
    env_cfg.set_scene(scene)
    env_cfg.episode_length_s = 30.0
    env = gym.make("DROID", cfg=env_cfg)

    # Reset twice for proper material loading
    obs, _ = env.reset()
    obs, _ = env.reset()

    sim_env = env.env
    wrist_cam = sim_env.scene["wrist_cam"]

    # Setup output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path("runs") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "tiptop_plan.pkl"

    print(f"[TiPToP-Sim] Scene: {scene}, Instruction: '{instruction}'")
    print(f"[TiPToP-Sim] Extracting initial observation...")

    # Extract observation from simulation
    sim_obs = extract_sim_observation(obs, wrist_cam)

    print(f"[TiPToP-Sim] Running TiPToP perception...")
    
    # Run TiPToP perception
    tamp_env, all_surfaces, grasps = asyncio.run(
        run_tiptop_perception_sim(
            rgb=sim_obs["external_rgb"],  # Use external camera for perception
            depth=sim_obs["depth"],
            intrinsics=sim_obs["intrinsics"],
            world_from_cam=sim_obs["world_from_cam"],
            task_instruction=instruction,
            joint_positions=sim_obs["joint_positions"],
        )
    )

    print(f"[TiPToP-Sim] Running cuTAMP planning...")
    
    # Run cuTAMP planning
    cutamp_plan = run_tiptop_planning(
        env=tamp_env,
        all_surfaces=all_surfaces,
        grasps=grasps,
        q_init=sim_obs["joint_positions"],
        save_dir=output_dir,
    )

    if cutamp_plan is None:
        print("[TiPToP-Sim] ERROR: Planning failed!")
        env.close()
        simulation_app.close()
        return

    # Save the plan
    with open(plan_path, "wb") as f:
        pickle.dump(cutamp_plan, f)
    print(f"[TiPToP-Sim] Saved plan to {plan_path}")

    # Now execute using the CutampJointPosClient
    print(f"[TiPToP-Sim] Executing plan in simulation...")
    
    client = CutampJointPosClient(
        file_dir=str(output_dir),
        file_name="tiptop_plan.pkl"
    )
    task_checker = get_checker(scene, vlm=False)

    video = []
    max_steps = env.env.max_episode_length

    with torch.no_grad():
        for ep in range(episodes):
            task_completed = False
            for i in tqdm(range(max_steps), desc=f"Episode {ep+1}/{episodes}"):
                ret = client.infer(obs, instruction)
                
                if not headless:
                    cv2.imshow("Execution", cv2.cvtColor(ret["viz"], cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
                
                video.append(ret["viz"])
                action = torch.tensor(ret["action"])[None]
                obs, _, term, trunc, _ = env.step(action)

                if i % 30 == 0 and not task_completed:
                    task_completed = task_checker.check(env.env, obs)
                    if task_completed:
                        print("[TiPToP-Sim] TASK COMPLETED!")
                        term = True

                if term or trunc:
                    break

            client.reset()
            mediapy.write_video(
                output_dir / f"tiptop_scene{scene}_ep{ep}.mp4",
                video,
                fps=15,
            )
            video = []

    print(f"[TiPToP-Sim] Results saved to {output_dir}")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    tyro.cli(main)