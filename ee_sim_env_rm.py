import numpy as np
import collections
import os

from constants import DT, XML_DIR
from constants import PUPPET_GRIPPER_POSITION_CLOSE
from constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN

from utils import sample_box_pose, sample_insertion_pose
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

import IPython
e = IPython.embed
START_ARM_POSE = [0 ,1.067 ,0.5233 ,0 ,1.57 ,0]

def make_ee_sim_env(task_name):
    """
    Environment for simulated robot bi-manual manipulation, with end-effector control.
    Action space:      [right_arm_pose (7),            # position and quaternion for end effector
                        right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ left_arm_qpos (6),         # absolute joint position
                                        left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                        right_arm_qpos (6),         # absolute joint position
                                        right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                        left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                        right_arm_qvel (6),         # absolute joint velocity (rad)
                                        right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"main": (480x640x3)}        # h, w, c, dtype='uint8'
    """
    if 'sim_grasp_cube_ur' in task_name:
        xml_path = os.path.join('./test/RMxml/bimanual_viperx_ee_transfer_cube_rm.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = GraspCubeEETask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    else:
        raise NotImplementedError
    return env

class BimanualViperXEETask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        a_len = len(action) 
        action_left = action[:a_len]

        # set mocap position and quat
        # left
        np.copyto(physics.data.mocap_pos[0], action_left[:3])
        np.copyto(physics.data.mocap_quat[0], action_left[3:7])

    def initialize_robots(self, physics):
        # reset joint position
        physics.named.data.qpos[:6] = START_ARM_POSE

        # reset mocap to align with end effector
        # to obtain these numbers:
        # (1) make an ee_sim env and reset to the same start_pose
        # (2) get env._physics.named.data.xpos['vx300s_left/gripper_link']
        #     get env._physics.named.data.xquat['vx300s_left/gripper_link']
        #     repeat the same for right side
        np.copyto(physics.data.mocap_pos[0], [0.31718881, 0.5, 0.29525084])
        np.copyto(physics.data.mocap_quat[0], [1, 0, 0, 0])

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        left_qpos_raw = qpos_raw[:6]
        left_arm_qpos = left_qpos_raw[:6]
        return np.concatenate([left_arm_qpos])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        left_qvel_raw = qvel_raw[:6]
        left_arm_qvel = left_qvel_raw[:6]
        return np.concatenate([left_arm_qvel])

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_observation(self, physics):
        # note: it is important to do .copy()
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
        obs['images']['vis'] = physics.render(height=480, width=640, camera_id='front_close')
        # used in scripted policy to obtain starting pose
        obs['mocap_pose_left'] = np.concatenate([physics.data.mocap_pos[0], physics.data.mocap_quat[0]]).copy()
        return obs

    def get_reward(self, physics):
        raise NotImplementedError

class GraspCubeEETask(BimanualViperXEETask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize box position
        cube_pose = sample_box_pose()
        box_start_idx = physics.model.name2id('red_box_joint', 'joint')
        np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7], cube_pose)
        # print(f"randomized cube position to {cube_position}")

        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[6:]
        return env_state
    
    def get_reward(self, physics):
        # 获取 "red_box" 和 "rm_65/gripper_link" 的位置
        
        gripper_pos = physics.named.data.xpos['rm_65/gripper_link']
        red_box_pos = physics.named.data.xpos['box']

        # 计算 gripper 和 red_box 的水平距离和高度差
        horizontal_distance = np.linalg.norm(gripper_pos[:2] - red_box_pos[:2])  # x, y 平面距离
        height_difference = gripper_pos[2] - red_box_pos[2]  # z 轴高度差

        # 奖励逻辑
        reward = 0
        if horizontal_distance < 0.05:  # 水平距离小于 5 厘米
            reward = 1
            if 0.09 <= height_difference <= 0.11:  # 高度差在 0.1 米 ± 1 厘米范围内
                reward = 4
            elif 0.05 <= height_difference < 0.09 or 0.11 < height_difference <= 0.15:  # 高度差稍微偏离
                reward = 2
        return reward
