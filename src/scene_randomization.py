"""Shared scene randomization logic for eval scripts.

Usage:
    from src.scene_randomization import SCENE_CONFIG, randomize_objects

Both tiptop_rand_eval.py and run_eval.py use the same SCENE_CONFIG and RNG
so that identical seeds produce identical object layouts across models.
"""

import numpy as np
import torch


# Per-scene configuration: which objects to randomize and table bounds.
SCENE_CONFIG = {
    1: {
        "instruction": "put the cube in the bowl",
        "objects": ["rubiks_cube", "_24_bowl"],
        "table_x": (0.35, 0.55),
        "table_y": (-0.20, 0.20),
        "min_dist": 0.12,
    },
    2: {
        "instruction": "put the can in the mug",
        "objects": ["_10_potted_meat_can", "_25_mug"],
        "table_x": (0.35, 0.55),
        "table_y": (-0.20, 0.20),
        "min_dist": 0.12,
    },
    3: {
        "instruction": "put banana in the bin",
        "objects": ["_11_banana", "small_KLT_visual_collision"],
        "table_x": (0.35, 0.55),
        "table_y": (-0.20, 0.20),
        "min_dist": 0.15,
    },
    4: {
        "instruction": "put the meat can on the sugar box",
        "objects": ["_10_potted_meat_can", "_04_sugar_box"],
        "table_x": (0.35, 0.55),
        "table_y": (-0.20, 0.20),
        "min_dist": 0.12,
    },
    5: {
        "instruction": "put three cubes into the bowl",
        "objects": ["blue_block", "green_block", "red_block", "yellow_block", "basic_block", "_24_bowl"],
        "table_x": (0.30, 0.65),
        "table_y": (-0.30, 0.30),
        "min_dist": 0.10,
    },
    6: {
        "instruction": "stack the cubes",
        "objects": ["dex_cube_instanceable", "dex_cube_instanceable_01", "dex_cube_instanceable_02"],
        "table_x": (0.30, 0.55),
        "table_y": (-0.25, 0.25),
        "min_dist": 0.10,
    },
}


def sample_positions(rng, num_objects, table_x, table_y, min_dist,
                     per_object_bounds=None, max_attempts=1000):
    """Sample non-overlapping (x, y) positions on the table.

    Args:
        per_object_bounds: Optional list of (table_x, table_y) tuples per object.
            If provided, each object uses its own bounds. None entries use the defaults.
    """
    for _ in range(max_attempts):
        positions = []
        for i in range(num_objects):
            if per_object_bounds and per_object_bounds[i] is not None:
                bx, by = per_object_bounds[i]
            else:
                bx, by = table_x, table_y
            x = rng.uniform(bx[0], bx[1])
            y = rng.uniform(by[0], by[1])
            positions.append((x, y))
        # Check all pairwise distances
        ok = True
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.sqrt((positions[i][0] - positions[j][0]) ** 2 +
                               (positions[i][1] - positions[j][1]) ** 2)
                if dist < min_dist:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return positions
    raise RuntimeError(f"Could not find valid positions after {max_attempts} attempts")


def randomize_objects(env, object_names, rng, table_x, table_y, min_dist,
                      object_bounds=None, **kwargs):
    """Randomize object (x, y) positions on the table, keeping original z and orientation.

    Args:
        object_bounds: Optional dict mapping object name to {"table_x": ..., "table_y": ...}
            for per-object placement bounds (e.g. to keep task objects in the wrist camera).
    """
    env_origins = env.env.scene.env_origins

    # Build per-object bounds list
    per_object_bounds = None
    if object_bounds:
        per_object_bounds = []
        for name in object_names:
            if name in object_bounds:
                b = object_bounds[name]
                per_object_bounds.append((b["table_x"], b["table_y"]))
            else:
                per_object_bounds.append(None)

    # Read original z heights and orientations
    obj_handles = []
    for name in object_names:
        obj = env.env.scene[name]
        orig_pos = (obj.data.root_pos_w - env_origins)[0].cpu().numpy()
        orig_quat = obj.data.root_quat_w[0].cpu().numpy()
        obj_handles.append((obj, orig_pos[2], orig_quat))

    # Sample new (x, y) positions
    positions = sample_positions(rng, len(object_names), table_x, table_y, min_dist,
                                 per_object_bounds=per_object_bounds)

    # Write new positions
    origin = env_origins[0].cpu().numpy()
    for (obj, z, quat), (x, y) in zip(obj_handles, positions):
        root_state = obj.data.default_root_state.clone()
        root_state[0, :3] = torch.tensor([x + origin[0], y + origin[1], z + origin[2]],
                                         dtype=torch.float32, device=obj.device)
        root_state[0, 3:7] = torch.tensor(quat, dtype=torch.float32, device=obj.device)
        root_state[0, 7:] = 0.0  # zero velocity
        obj.write_root_state_to_sim(root_state)

    return positions
