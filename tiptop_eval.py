"""
Example script for running TiPToP rollouts in the DROID environment.

Usage:

First, make sure you download the simulation assets and unpack them into the root directory of this package.

Finally, run the evaluation script:

python tiptop_eval.py --episodes 10 --headless
"""

import tyro
import argparse
import torch
import cv2
import asyncio
import numpy as np
import aiohttp
import sys
import types
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation

CUROBO_PATH = "/home/ubuntu/curobo/src"
if CUROBO_PATH not in sys.path:
    sys.path.insert(0, CUROBO_PATH)

CUTAMP_PATH = "/home/ubuntu/cutamp"
if CUTAMP_PATH not in sys.path:
    sys.path.insert(0, CUTAMP_PATH)

TIPTOP_PATH = "/home/ubuntu/tiptop-robot"
if TIPTOP_PATH not in sys.path:
    sys.path.insert(0, TIPTOP_PATH)

OPENPI_CLIENT_PATH = "/home/ubuntu/openpi/packages/openpi-client/src"
if OPENPI_CLIENT_PATH not in sys.path:
    sys.path.insert(0, OPENPI_CLIENT_PATH)

CONDA_TIPTOP_SITE_PACKAGES = "/home/ubuntu/miniconda3/envs/tiptop/lib/python3.10/site-packages"
if CONDA_TIPTOP_SITE_PACKAGES not in sys.path:
    sys.path.append(CONDA_TIPTOP_SITE_PACKAGES)

try:
    import gymnasium as gym
except ImportError:
    gym = types.ModuleType("gymnasium")
    _gym_registry = {}

    def _register(id, entry_point, kwargs=None, **_):
        _gym_registry[id] = {"entry_point": entry_point, "kwargs": kwargs or {}}

    def _make(id, **kwargs):
        if id not in _gym_registry:
            raise ValueError(f"Unknown env id: {id}")
        spec = _gym_registry[id]
        return spec["entry_point"](**spec["kwargs"], **kwargs)

    gym.register = _register
    gym.make = _make
    sys.modules["gymnasium"] = gym

try:
    import setuptools_scm  # noqa: F401
except ImportError:
    class _SetuptoolsScmFallback:
        @staticmethod
        def get_version(*args, **kwargs):
            return "v0.0.0"

    sys.modules["setuptools_scm"] = _SetuptoolsScmFallback()

from openpi_client import image_tools

from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from cutamp.algorithm import run_cutamp
from cutamp.config import TAMPConfiguration
from cutamp.constraint_checker import ConstraintChecker
from cutamp.cost_reduction import CostReducer
from cutamp.scripts.utils import default_constraint_to_mult, default_constraint_to_tol
from cutamp.task_planning.constraints import StablePlacement
from tiptop.config import tiptop_cfg
from tiptop.motion_planning import get_ik_solver, get_motion_gen
from tiptop.perception.m2t2 import generate_grasps_async
from tiptop.perception.utils import depth_to_xyz, get_o3d_pcd
from tiptop.perception_wrapper import detect_and_segment
from tiptop.tiptop_run import create_tamp_environment, process_scene_geometry, VOXEL_DOWNSAMPLE_SIZE
from tiptop.workspace import workspace_cuboids


class TipTopPlanner:
    def __init__(self, num_particles: int = 256, max_planning_time: float = 30.0, opt_steps_per_skeleton: int = 500):
        collision_activation_distance = 0.0
        self.config = TAMPConfiguration(
            num_particles=num_particles,
            max_loop_dur=max_planning_time,
            num_opt_steps=opt_steps_per_skeleton,
            m2t2_grasps=True,
            prop_satisfying_break=0.1,
            robot=tiptop_cfg().robot.type,
            curobo_plan=True,
            warmup_ik=False,
            warmup_motion_gen=False,
            num_initial_plans=10,
            cache_subgraphs=True,
            world_activation_distance=collision_activation_distance,
            movable_activation_distance=0.01,
            time_dilation_factor=tiptop_cfg().robot.time_dilation_factor,
            placement_check="obb",
            placement_shrink_dist=0.01,
            enable_visualizer=False,
            coll_sphere_radius=0.008,
        )
        num_spheres = self.config.coll_n_spheres
        cuboids = [
            *workspace_cuboids(),
            Cuboid(name="table", dims=[0.01, 0.01, 0.01], pose=[99.99, 99.9, 99.9, 1.0, 0.0, 0.0, 0.0]),
        ]
        world_cfg = WorldConfig(cuboid=cuboids)
        self.ik_solver = get_ik_solver(world_cfg, num_particles)
        self.motion_gen = get_motion_gen(
            world_cfg, collision_activation_distance=collision_activation_distance, num_spheres=num_spheres
        )
        self.tensor_args = TensorDeviceType()
        self.constraint_to_mult = default_constraint_to_mult.copy()
        self.constraint_to_tol = default_constraint_to_tol.copy()
        self.cost_reducer = CostReducer(self.constraint_to_mult.copy())

    @staticmethod
    def _world_from_cam(pos_w: np.ndarray, quat_w_ros: np.ndarray) -> np.ndarray:
        quat_xyzw = np.array([quat_w_ros[1], quat_w_ros[2], quat_w_ros[3], quat_w_ros[0]])
        rot = Rotation.from_quat(quat_xyzw).as_matrix()
        world_from_cam = np.eye(4, dtype=np.float32)
        world_from_cam[:3, :3] = rot
        world_from_cam[:3, 3] = pos_w
        return world_from_cam

    async def _predict_depth_and_grasps_sim(
        self,
        session: aiohttp.ClientSession,
        depth_m: np.ndarray,
        rgb: np.ndarray,
        K: np.ndarray,
        world_from_cam: np.ndarray,
        use_m2t2: bool = True,
    ) -> dict:
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        xyz_map = depth_to_xyz(depth_m, fx, fy, cx, cy)
        xyz_map = xyz_map @ world_from_cam[:3, :3].T + world_from_cam[:3, 3]
        rgb_map = rgb.astype(np.float32) / 255.0
        pcd = await asyncio.to_thread(get_o3d_pcd, xyz_map, rgb_map, VOXEL_DOWNSAMPLE_SIZE)
        xyz_downsampled = np.asarray(pcd.points)
        rgb_downsampled = np.asarray(pcd.colors)
        if use_m2t2:
            grasps = await generate_grasps_async(
                session, tiptop_cfg().perception.m2t2.url, scene_xyz=xyz_downsampled, scene_rgb=rgb_downsampled
            )
        else:
            grasps = {}

        return {
            "depth_map": depth_m,
            "xyz_map": xyz_map,
            "rgb_map": rgb_map,
            "xyz_downsampled": xyz_downsampled,
            "rgb_downsampled": rgb_downsampled,
            "pcd_downsampled": pcd,
            "grasps": grasps,
        }

    async def plan(
        self,
        session: aiohttp.ClientSession,
        instruction: str,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        K: np.ndarray,
        world_from_cam: np.ndarray,
        q_curr: np.ndarray,
        output_dir: Path,
    ):
        depth_results, detection_results = await asyncio.gather(
            self._predict_depth_and_grasps_sim(session, depth_m, rgb, K, world_from_cam),
            detect_and_segment(rgb, instruction),
        )
        processed_scene, _ = process_scene_geometry(
            depth_results["xyz_map"],
            depth_results["rgb_map"],
            detection_results["masks"],
            detection_results["bboxes"],
            depth_results["grasps"],
        )
        env, all_surfaces = create_tamp_environment(
            processed_scene.object_meshes, processed_scene.table_cuboid, detection_results["grounded_atoms"]
        )
        for surface in all_surfaces:
            self.constraint_to_tol[StablePlacement.type][f"{surface.name}_in_xy"] = 1e-2
            self.constraint_to_tol[StablePlacement.type][f"{surface.name}_support"] = 1e-2
            self.constraint_to_mult[StablePlacement.type][f"{surface.name}_support"] = 1.0
        constraint_checker = ConstraintChecker(self.constraint_to_tol)

        cutamp_plan, _ = run_cutamp(
            env,
            self.config,
            self.cost_reducer,
            constraint_checker,
            q_init=q_curr,
            ik_solver=self.ik_solver,
            grasps=processed_scene.grasps,
            motion_gen=self.motion_gen,
            experiment_dir=output_dir,
        )
        return cutamp_plan


class TipTopClient:
    def __init__(self, wrist_cam, output_root: Path):
        self.wrist_cam = wrist_cam
        self.output_root = output_root
        self.planner = TipTopPlanner()
        self.actions = []
        self.action_index = 0
        self.gripper_state = 0.0
        self._planned = False

    def reset(self):
        self.actions = []
        self.action_index = 0
        self.gripper_state = 0.0
        self._planned = False

    def _extract_joint_positions(self, obs: dict) -> np.ndarray:
        joint_pos = obs["policy"]["arm_joint_pos"]
        if isinstance(joint_pos, torch.Tensor):
            joint_pos = joint_pos.clone().detach().cpu().numpy()
        if joint_pos.ndim > 1:
            joint_pos = joint_pos[0]
        return joint_pos

    def _extract_gripper(self, obs: dict) -> float:
        gripper_pos = obs["policy"]["gripper_pos"]
        if isinstance(gripper_pos, torch.Tensor):
            gripper_pos = gripper_pos.clone().detach().cpu().numpy()
        gripper_pos = np.asarray(gripper_pos)
        if gripper_pos.ndim > 0:
            gripper_pos = gripper_pos[0]
        return float(gripper_pos)

    def _build_actions(self, cutamp_plan, obs: dict):
        if cutamp_plan is None:
            return
        current_q = self._extract_joint_positions(obs)
        self.gripper_state = self._extract_gripper(obs)
        hold_steps = 10
        for action_dict in cutamp_plan:
            if action_dict["type"] == "gripper":
                action = action_dict["action"]
                self.gripper_state = 1.0 if action == "close" else 0.0
                for _ in range(hold_steps):
                    self.actions.append(np.concatenate([current_q, np.array([self.gripper_state])]))
            elif action_dict["type"] == "trajectory":
                waypoints = action_dict["plan"].position.cpu().numpy()
                for q in waypoints:
                    current_q = q
                    self.actions.append(np.concatenate([q, np.array([self.gripper_state])]))
            else:
                raise ValueError(f"Unknown action type: {action_dict['type']}")

    def _plan_once(self, obs: dict, instruction: str):
        rgb = obs["policy"]["wrist_cam"][0].clone().detach().cpu().numpy()
        rgb = np.asarray(rgb, dtype=np.uint8)
        depth_m = self.wrist_cam.data.output["distance_to_image_plane"][0].clone().detach().cpu().numpy()
        depth_m = np.asarray(depth_m, dtype=np.float32)
        K = self.wrist_cam.data.intrinsic_matrices[0].cpu().numpy()
        pos_w = self.wrist_cam.data.pos_w[0].cpu().numpy()
        quat_w_ros = self.wrist_cam.data.quat_w_ros[0].cpu().numpy()
        world_from_cam = self.planner._world_from_cam(pos_w, quat_w_ros)
        q_curr = self._extract_joint_positions(obs)

        output_dir = self.output_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir.mkdir(parents=True, exist_ok=True)

        connector = aiohttp.TCPConnector(limit=10, force_close=True)
        async def _run():
            async with aiohttp.ClientSession(connector=connector) as session:
                return await self.planner.plan(
                    session, instruction, rgb, depth_m, K, world_from_cam, q_curr, output_dir
                )

        cutamp_plan = asyncio.run(_run())
        self._build_actions(cutamp_plan, obs)
        self._planned = True

    def infer(self, obs: dict, instruction: str) -> dict:
        if not self._planned:
            self._plan_once(obs, instruction)

        if self.action_index < len(self.actions):
            action = self.actions[self.action_index]
            self.action_index += 1
        else:
            joint_pos = self._extract_joint_positions(obs)
            action = np.concatenate([joint_pos, np.array([self.gripper_state])])
        action = np.asarray(action, dtype=np.float32)

        right_image = obs["policy"]["external_cam"][0].clone().detach().cpu().numpy()
        wrist_image = obs["policy"]["wrist_cam"][0].clone().detach().cpu().numpy()
        img1 = image_tools.resize_with_pad(right_image, 224, 224)
        img2 = image_tools.resize_with_pad(wrist_image, 224, 224)
        viz = np.concatenate([img1, img2], axis=1)
        return {"action": action, "viz": viz, "right_image": right_image, "wrist_image": wrist_image}


def main(
        episodes:int = 10,
        headless: bool = True,
        scene: int = 1,
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
    import src.environments # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from src.inference.termination_checker import get_checker


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
            instruction = "put the meat can on the sugar box"
        case 5:
            instruction = "rearrange the cubes so that they spell 'REX'"
        case 6:
            instruction = "stack all the cubes on top of each other"
        case _:
            raise ValueError(f"Scene {scene} not supported")
        
    env_cfg.set_scene(scene)
    env_cfg.episode_length_s = 30.0 # LENGTH OF EPISODE
    env = gym.make("DROID", cfg=env_cfg)

    obs, _ = env.reset()
    obs, _ = env.reset() # need second render cycle to get correctly loaded materials

    sim_env = env.env
    wrist_cam = sim_env.scene["wrist_cam"]
    output_root = Path("tiptop_outputs")
    client = TipTopClient(wrist_cam=wrist_cam, output_root=output_root)
    task_checker = get_checker(scene, vlm=False)

    video_dir = Path("runs") / datetime.now().strftime("%Y-%m-%d") / datetime.now().strftime("%H-%M-%S")
    video_dir.mkdir(parents=True, exist_ok=True)
    video = []
    right_video = []
    wrist_video = []
    ep = 0
    max_steps = env.env.max_episode_length
    with torch.no_grad():
        for ep in range(episodes):
            task_completed = False
            for i in tqdm(range(max_steps), desc=f"Episode {ep+1}/{episodes}"):
                ret = client.infer(obs, instruction)
                if not headless:
                    cv2.imshow("Right Camera", cv2.cvtColor(ret["viz"], cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
                video.append(ret["viz"])
                right_video.append(ret["right_image"])
                wrist_video.append(ret["wrist_image"])
                action = torch.tensor(ret["action"])[None]
                obs, _, term, trunc, _ = env.step(action)

                if i % 30 == 0 and not task_completed:
                    task_completed = task_checker.check(env.env, obs)
                    if task_completed:
                        print("TASK COMPLETED")
                        term = True

                if term or trunc:
                    break

            client.reset()
            mediapy.write_video(video_dir / f"tiptop_scene{scene}_ep{ep}.mp4", video, fps=15)
            # # added right and wrist videos
            # mediapy.write_video(
            #     video_dir / f"{policy}_scene{scene}_ep{ep}_right.mp4",
            #     right_video,
            #     fps=15,
            # )
            # mediapy.write_video(
            #     video_dir / f"{policy}_scene{scene}_ep{ep}_wrist.mp4",
            #     wrist_video,
            #     fps=15,
            # )
            video = []
            # # reset right and wrist videos
            # right_video = []
            # wrist_video = []

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    args = tyro.cli(main)
try:
    import mediapy
except ImportError:
    class _MediaPyShim:
        @staticmethod
        def write_video(*args, **kwargs):
            return None

    mediapy = _MediaPyShim()
