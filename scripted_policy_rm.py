import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

from constants import SIM_TASK_CONFIGS
from ee_sim_env_rm import make_ee_sim_env

import IPython
e = IPython.embed


class BasePolicy:
    def __init__(self, inject_noise=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.left_trajectory = None

    def generate_trajectory(self, ts_first):
        raise NotImplementedError

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        curr_xyz = curr_waypoint['xyz']
        curr_quat = curr_waypoint['quat']
        next_xyz = next_waypoint['xyz']
        next_quat = next_waypoint['quat']
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        quat = curr_quat + (next_quat - curr_quat) * t_frac
        return xyz, quat

    def __call__(self, ts):
        # generate trajectory at first timestep, then open-loop execution
        if self.step_count == 0:
            self.generate_trajectory(ts)

        # obtain left and right waypoints
        if self.left_trajectory[0]['t'] == self.step_count:
            self.curr_left_waypoint = self.left_trajectory.pop(0)
        next_left_waypoint = self.left_trajectory[0]

        # interpolate between waypoints to obtain current pose and gripper command
        left_xyz, left_quat = self.interpolate(self.curr_left_waypoint, next_left_waypoint, self.step_count)

        # Inject noise
        if self.inject_noise:
            scale = 0.01
            left_xyz = left_xyz + np.random.uniform(-scale, scale, left_xyz.shape)

        action_left = np.concatenate([left_xyz, left_quat])

        self.step_count += 1
        return np.concatenate([action_left])


class PickAndTransferPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        box_info = np.array(ts_first.observation['env_state'])
        box_xyz = box_info[:3]
        box_quat = box_info[3:]
        # print(f"Generate trajectory for {box_xyz=}")

        gripper_pick_quat = Quaternion(init_mocap_pose_left[3:])
        gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)
        '''
        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:]}, # sleep
            {"t": 90, "xyz": box_xyz + np.array([0, 0, 0.2]), "quat": gripper_pick_quat.elements}, # approach the cube
            {"t": 130, "xyz": box_xyz + np.array([0, 0, +0.01]), "quat": gripper_pick_quat.elements}, # go down
            {"t": 170, "xyz": box_xyz + np.array([0, 0, +0.01]), "quat": gripper_pick_quat.elements} # close gripper
        ]
        '''
        gripper_pick_quat_grasp =  Quaternion(axis=[0.0, 1.0, 0.0], degrees=180)
        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat":gripper_pick_quat_grasp.elements}, # sleep
            {"t": 90, "xyz": box_xyz + np.array([0, 0, 0.2]), "quat": gripper_pick_quat_grasp.elements}, # approach the cube
            {"t": 130, "xyz": box_xyz + np.array([0, 0, +0.01]), "quat":gripper_pick_quat_grasp.elements}, # go down
            {"t": 170, "xyz": box_xyz + np.array([0, 0, +0.01]), "quat": gripper_pick_quat_grasp.elements} # close gripper
        ]

def test_policy(task_name):
    # example rolling out pick_and_transfer policy
    onscreen_render = True
    inject_noise = False

    # setup the environment
    episode_len = 100
    if 'sim_grasp_cube_ur' in task_name:
        env = make_ee_sim_env('sim_grasp_cube_ur')
    else:
        raise NotImplementedError

    for episode_idx in range(2):
        ts = env.reset()
        episode = [ts]
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images']['angle'])
            plt.ion()

        policy = PickAndTransferPolicy(inject_noise)
        for step in range(episode_len):
            action = policy(ts)
            ts = env.step(action)
            episode.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images']['angle'])
                plt.pause(0.02)
        plt.close()

        episode_return = np.sum([ts.reward for ts in episode[1:]])
        if episode_return > 0:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")


if __name__ == '__main__':
    test_task_name = 'sim_grasp_cube_ur'
    test_policy(test_task_name)

