import logging
import time
import numpy as np
from typing import List, Optional, Callable
import torch

from isaaclab.envs import ManagerBasedRLEnv

class IsaacSimRobotClient:

    def __init__(self, env: ManagerBasedRLEnv):
        self.env = env

    def get_joint_positions(self) -> List[float]:
        pass

    def open_gripper(self):
        pass

    def close_gripper(self):
        pass

    def execute_joint_impedance_path(self): # execute a joint trajectory 
        pass

    def close(self): # disconnect from the robot and gripper
        pass