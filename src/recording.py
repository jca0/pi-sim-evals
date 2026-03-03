"""Utilities for recording sim episodes in raw DROID format.

Produces data compatible with openpi's convert_droid_data_to_lerobot.py:
  - trajectory.h5 (robot state + actions + camera metadata)
  - recordings/MP4/<serial>.mp4 (one video per camera)
  - metadata_<uuid>.json (episode metadata)
  - aggregated-annotations-030724.json (language instructions)
"""

import json
from datetime import datetime
from pathlib import Path

import cv2
import h5py
import numpy as np

# Fake camera serial numbers for sim data (used by convert_droid_data_to_lerobot.py)
SIM_WRIST_CAM_SERIAL = "sim_wrist"
SIM_EXT1_CAM_SERIAL = "sim_ext1"
SIM_EXT2_CAM_SERIAL = "sim_ext2"


def save_episode_droid_format(
    episode_dir: Path,
    episode_uuid: str,
    instruction: str,
    ext_images: list[np.ndarray],
    wrist_images: list[np.ndarray],
    joint_positions: list[np.ndarray],
    gripper_positions: list[float],
    actions: list[np.ndarray],
    fps: int = 15,
):
    """Save one episode in raw DROID format (trajectory.h5 + MP4s + metadata JSON).

    This produces data compatible with convert_droid_data_to_lerobot.py.
    """
    episode_dir.mkdir(parents=True, exist_ok=True)
    T = min(len(joint_positions), len(actions))

    joint_pos_arr = np.array(joint_positions[:T], dtype=np.float64)  # (T, 7)
    gripper_pos_arr = np.array(gripper_positions[:T], dtype=np.float64)  # (T,)
    actions_arr = np.array(actions[:T], dtype=np.float64)  # (T, 8)

    # Compute joint velocities from consecutive joint positions: v[t] = (pos[t+1] - pos[t]) * fps
    # For the last timestep, repeat the previous velocity.
    joint_vel = np.diff(joint_pos_arr, axis=0) * fps  # (T-1, 7)
    joint_vel = np.concatenate([joint_vel, joint_vel[-1:]], axis=0)  # (T, 7)

    _write_trajectory_h5(episode_dir / "trajectory.h5", T, joint_vel, actions_arr, joint_pos_arr, gripper_pos_arr)
    _write_mp4s(episode_dir / "recordings" / "MP4", ext_images, wrist_images, fps)
    _write_metadata(episode_dir, episode_uuid, instruction, T)

    print(f"Saved DROID-format episode to {episode_dir} ({T} steps)")


def _write_trajectory_h5(
    path: Path,
    T: int,
    joint_vel: np.ndarray,
    actions_arr: np.ndarray,
    joint_pos_arr: np.ndarray,
    gripper_pos_arr: np.ndarray,
):
    with h5py.File(path, "w") as f:
        # Actions — conversion script uses action/joint_velocity + action/gripper_position
        f.create_dataset("action/joint_velocity", data=joint_vel)
        f.create_dataset("action/gripper_position", data=actions_arr[:, 7])
        f.create_dataset("action/joint_position", data=actions_arr[:, :7])

        # Observations
        f.create_dataset("observation/robot_state/joint_positions", data=joint_pos_arr)
        f.create_dataset("observation/robot_state/gripper_position", data=gripper_pos_arr)

        # Camera type: 0 = wrist, 1 = exterior (used by conversion to classify cameras)
        f.create_dataset(f"observation/camera_type/{SIM_WRIST_CAM_SERIAL}", data=np.zeros(T, dtype=np.int64))
        f.create_dataset(f"observation/camera_type/{SIM_EXT1_CAM_SERIAL}", data=np.ones(T, dtype=np.int64))
        f.create_dataset(f"observation/camera_type/{SIM_EXT2_CAM_SERIAL}", data=np.ones(T, dtype=np.int64))

        # Controller info — all steps are valid (no idle filtering)
        f.create_dataset("observation/controller_info/movement_enabled", data=np.ones(T, dtype=bool))

        # Timestamps (fake but required by load_trajectory)
        ts = np.arange(T, dtype=np.int64)
        for serial in [SIM_WRIST_CAM_SERIAL, SIM_EXT1_CAM_SERIAL, SIM_EXT2_CAM_SERIAL]:
            f.create_dataset(f"observation/timestamp/cameras/{serial}_frame_received", data=ts)


def _write_mp4s(mp4_dir: Path, ext_images: list[np.ndarray], wrist_images: list[np.ndarray], fps: int):
    mp4_dir.mkdir(parents=True, exist_ok=True)
    for serial, frames in [
        (SIM_WRIST_CAM_SERIAL, wrist_images),
        (SIM_EXT1_CAM_SERIAL, ext_images),
        (SIM_EXT2_CAM_SERIAL, ext_images),  # duplicate external cam for ext2
    ]:
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(mp4_dir / f"{serial}.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()


def _write_metadata(episode_dir: Path, episode_uuid: str, instruction: str, T: int):
    metadata = {
        "uuid": episode_uuid,
        "lab": "SIM",
        "user": "sim",
        "user_id": "sim",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "timestamp": datetime.now().strftime("%Y-%m-%d-%Hh-%Mm-%Ss"),
        "success": True,
        "robot_serial": "panda-sim",
        "trajectory_length": T,
        "wrist_cam_serial": SIM_WRIST_CAM_SERIAL,
        "ext1_cam_serial": SIM_EXT1_CAM_SERIAL,
        "ext2_cam_serial": SIM_EXT2_CAM_SERIAL,
        "current_task": instruction,
    }
    with open(episode_dir / f"metadata_{episode_uuid}.json", "w") as f:
        json.dump(metadata, f, indent=2)


def update_annotations(annotations_path: Path, episode_uuid: str, instruction: str):
    """Add/update the language annotation for an episode in the aggregated annotations file."""
    if annotations_path.exists():
        with open(annotations_path) as f:
            annotations = json.load(f)
    else:
        annotations = {}

    annotations[episode_uuid] = {
        "language_instruction1": instruction,
        "language_instruction2": instruction,
        "language_instruction3": instruction,
    }
    with open(annotations_path, "w") as f:
        json.dump(annotations, f, indent=2)
