import torch
import numpy as np

# Curobo imports
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

# DROID imports
from openpi_client import image_tools

class CuroboClient:
    def __init__(self, env, device="cuda:0"):
        self.env = env
        self.device = device
        
        self._init_curobo()
        
        self.state = "INIT"
        self.plan = None
        self.plan_idx = 0
        self.gripper_state = 0.0
        self.gripper_timer = 0
        
        self.cube_name = "rubiks_cube"
        self.bowl_name = "_24_bowl"
        

    def _init_curobo(self):
        robot_file = "franka.yml"
        world_file = "collision_table.yml" 
        
        tensor_args = TensorDeviceType(device=torch.device(self.device))
        
        robot_cfg_path = join_path(get_robot_configs_path(), robot_file)
        robot_cfg = load_yaml(robot_cfg_path)["robot_cfg"]
        
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            world_file,
            tensor_args,
            use_cuda_graph=True,
        )
        self.motion_gen = MotionGen(motion_gen_config)
        self.motion_gen.warmup()

    def _get_object_pose(self, name):
        if name is None:
            return None
        pos = self.env.env.scene[name].data.root_pos_w[0]
        quat = self.env.env.scene[name].data.root_quat_w[0] # (w, x, y, z)
        
        # Convert to Curobo Pose [x, y, z, w, x, y, z]
        return Pose(
            position=pos.unsqueeze(0), 
            quaternion=quat.unsqueeze(0)
        )

    def plan_to_pose(self, current_q, target_pose):
        if current_q.dim() == 1:
            start_state = current_q.unsqueeze(0) # Shape (1, 7)
        else:
            start_state = current_q # Shape (1, 7)
            
        js = JointState.from_position(start_state)
        
        # Plan trajectory
        result = self.motion_gen.plan_single(js, target_pose.clone())
        
        if result.success.item():
            return result.interpolated_plan
        else:
            print("Motion planning failed!")
            return None

    def reset(self):
        self.state = "INIT"
        self.plan = None
        self.plan_idx = 0
        self.gripper_state = 0.0
        self.gripper_timer = 0

    def visualize(self, obs):
        right_image = obs["policy"]["external_cam"][0].cpu().numpy()
        wrist_image = obs["policy"]["wrist_cam"][0].cpu().numpy()
        
        img1 = image_tools.resize_with_pad(right_image, 224, 224)
        img2 = image_tools.resize_with_pad(wrist_image, 224, 224)
        return np.concatenate([img1, img2], axis=1)

    def infer(self, obs, instruction):
        if isinstance(obs["policy"]["arm_joint_pos"], torch.Tensor):
             joint_pos = obs["policy"]["arm_joint_pos"].to(self.device).view(-1)
        else:
             joint_pos = torch.tensor(obs["policy"]["arm_joint_pos"], device=self.device).view(-1)
             
        action_q = joint_pos.clone()
        
        def get_action():
            act = torch.cat([action_q, torch.tensor([self.gripper_state], device=self.device)])
            return act.cpu().numpy()

        def align_with_cube(cube_pose, offset_z=0.0):
            target = cube_pose.clone()
            target.position[0, 2] += offset_z
            
            q_cube = cube_pose.quaternion[0] # (4,)
            w, x, y, z = q_cube[0], q_cube[1], q_cube[2], q_cube[3]
            
            q_target = torch.tensor([-x, w, z, -y], device=self.device)
            target.quaternion[0] = q_target
            return target

        # --- State Machine ---
        if self.state == "INIT":
            if self.cube_name:
                self.state = "PLAN_PICK"
            else:
                print("Cube not found, staying idle.")
        
        elif self.state == "PLAN_PICK":
            cube_pose = self._get_object_pose(self.cube_name)
            target = align_with_cube(cube_pose, offset_z=0.3)
            
            self.plan = self.plan_to_pose(joint_pos, target)
            if self.plan is not None:
                self.plan_idx = 0
                self.state = "EXECUTE_PRE_PICK"

        elif self.state == "EXECUTE_PRE_PICK":
            if self.plan is not None and self.plan_idx < self.plan.position.shape[0]:
                action_q = self.plan.position[self.plan_idx, :7]
                self.plan_idx += 1
            else:
                self.state = "PLAN_GRASP"
                
        elif self.state == "PLAN_GRASP":
            cube_pose = self._get_object_pose(self.cube_name)
            target = align_with_cube(cube_pose, offset_z=0.13)
            
            self.plan = self.plan_to_pose(joint_pos, target)
            if self.plan is not None:
                self.plan_idx = 0
                self.state = "EXECUTE_GRASP"

        elif self.state == "EXECUTE_GRASP":
            if self.plan is not None and self.plan_idx < self.plan.position.shape[0]:
                action_q = self.plan.position[self.plan_idx, :7]
                self.plan_idx += 1
            else:
                self.state = "CLOSE_GRIPPER"
                self.gripper_timer = 0

        elif self.state == "CLOSE_GRIPPER":
            self.gripper_state = 1.0 # Close
            self.gripper_timer += 1
            if self.gripper_timer > 20: # Wait for gripper to close
                self.state = "PLAN_LIFT"
        
        elif self.state == "PLAN_LIFT":
            cube_pose = self._get_object_pose(self.cube_name)
            target = align_with_cube(cube_pose, offset_z=0.3)
            
            self.plan = self.plan_to_pose(joint_pos, target)
            if self.plan is not None:
                self.plan_idx = 0
                self.state = "EXECUTE_LIFT"

        elif self.state == "EXECUTE_LIFT":
            if self.plan is not None and self.plan_idx < self.plan.position.shape[0]:
                action_q = self.plan.position[self.plan_idx, :7]
                self.plan_idx += 1
            else:
                self.state = "PLAN_PLACE"

        elif self.state == "PLAN_PLACE":
            if self.bowl_name:
                bowl_pose = self._get_object_pose(self.bowl_name)
                target = bowl_pose.clone()
                target.position[0, 2] += 0.4 # Above bowl
                target.quaternion[0] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device)
                
                self.plan = self.plan_to_pose(joint_pos, target)
                if self.plan is not None:
                    self.plan_idx = 0
                    self.state = "EXECUTE_PLACE"
            else:
                self.state = "DONE"

        elif self.state == "EXECUTE_PLACE":
            if self.plan is not None and self.plan_idx < self.plan.position.shape[0]:
                action_q = self.plan.position[self.plan_idx, :7]
                self.plan_idx += 1
            else:
                self.state = "OPEN_GRIPPER"
                self.gripper_timer = 0

        elif self.state == "OPEN_GRIPPER":
            self.gripper_state = 0.0 # Open
            self.gripper_timer += 1
            if self.gripper_timer > 20:
                self.state = "DONE"

        return {
            "action": get_action(),
            "viz": self.visualize(obs)
        }