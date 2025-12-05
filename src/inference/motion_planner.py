import numpy as np
import torch
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation as R

try:
    # CuRobo Imports
    from curobo.geom.types import WorldConfig, Cuboid, Mesh
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.types.robot import RobotConfig
    from curobo.types.state import JointState
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
    CUROBO_AVAILABLE = True
except ImportError:
    print("Wait: CuRobo not installed. Please install with `pip install curobo`")
    CUROBO_AVAILABLE = False

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from .abstract_client import InferenceClient