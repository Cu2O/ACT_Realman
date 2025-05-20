import numpy as np
import os
import collections
import matplotlib.pyplot as plt
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

from constants import DT, XML_DIR
from constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
from constants import MASTER_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN

import IPython
e = IPython.embed
START_ARM_POSE = [-4.74410956e-01, -2.72182426e-01,  2.04231873e+00 ,-8.91448876e-04, 1.36208805e+00  ,1.09321004e+00]
BOX_POSE = [None] # to be changed from outside

def make_sim_env_rm(task_name):
    """
    Environment for simulated robot bi-manual manipulation, with joint position control
    Action space:      [left_arm_qpos (6),             # absolute joint position
    Observation space: {"qpos": Concat[ left_arm_qpos (6),         # absolute joint position
                        "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                        "images": {"main": (480x640x3)}        # h, w, c, dtype='uint8'
    """
    if 'sim_grasp_cube_ur' in task_name:
        xml_path = os.path.join('./test/RMxml/bimanual_viperx_transfer_cube_rm.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = GraspCubeTask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    else:
        raise NotImplementedError
    return env

class BimanualViperXTask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        left_arm_action = action[:6]
        # print('Target joint positions:', left_arm_action)
        # 将目标位置赋值给控制输入
        np.copyto(physics.data.ctrl[:6], left_arm_action)
        # 调用父类的 before_step 方法
        super().before_step(action, physics)

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
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
        obs['images']['vis'] = physics.render(height=480, width=640, camera_id='front_close')

        return obs

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        raise NotImplementedError

class GraspCubeTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        with physics.reset_context():
            physics.named.data.qpos[:6] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7:] = BOX_POSE[0]
            # print(f"{BOX_POSE=}")
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

def get_action(master_bot_left, master_bot_right):
    action = np.zeros(6)
    # arm action
    action[:6] = master_bot_left.dxl.joint_states.position[:6]
    return action

def test_sim_teleop():
    """ Testing teleoperation in sim with ALOHA. Requires hardware and ALOHA repo to work. """
    from interbotix_xs_modules.arm import InterbotixManipulatorXS

    BOX_POSE[0] = [0.2, 0.5, 0.05, 1, 0, 0, 0]

    # source of data
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_left', init_node=True)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_right', init_node=False)

    # setup the environment
    env = make_sim_env_rm('sim_transfer_cube')
    ts = env.reset()
    episode = [ts]
    # setup plotting
    ax = plt.subplot()
    plt_img = ax.imshow(ts.observation['images']['angle'])
    plt.ion()

    for t in range(1000):
        action = get_action(master_bot_left, master_bot_right)
        ts = env.step(action)
        episode.append(ts)

        plt_img.set_data(ts.observation['images']['angle'])
        plt.pause(0.02)


if __name__ == '__main__':
    test_sim_teleop()

