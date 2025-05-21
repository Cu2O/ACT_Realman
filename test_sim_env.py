import numpy as np
import os
from dm_control import mujoco
from dm_control.rl import control
from dm_control.rl.control import Task

XML_PATH = './test/RMxml/bimanual_viperx_transfer_cube_rm.xml'

def test_position_control():
    """
    Test position control for the robot using the specified XML file.
    """
    # Load the XML file
    physics = mujoco.Physics.from_xml_path(XML_PATH)
    start_pos = np.array([-0.5, -0.3, 2.0, 0.0, 1.3, 1.1])
    # Define a simple task for testing
    class SimpleTask(Task):
        def __init__(self):
            super().__init__()

        def initialize_episode(self, physics):
            # Set initial joint positions
            physics.named.data.qpos[:6] = start_pos
            np.copyto(physics.data.ctrl, physics.named.data.qpos[:6])

        def before_step(self, action, physics):
            # Apply the action (target joint positions)
            # np.copyto(physics.data.ctrl[:6],action[0:6])
            np.copyto(physics.data.qpos[:6],action[0:6])
            np.copyto(physics.data.qvel[:6],np.zeros(6))

        def get_observation(self, physics):
            # Return joint positions and velocities as observations
            return {
                'qpos': physics.data.qpos[:6].copy(),
                'qvel': physics.data.qvel[:6].copy()
            }

        def get_reward(self, physics):
            # Dummy reward function
            return 0

        def action_spec(self, physics):
            # Define the action space (same size as the number of joints)
            return control.ActionSpec(shape=(6,), minimum=-np.inf, maximum=np.inf)

    # Create the environment
    task = SimpleTask()
    env = control.Environment(physics, task, time_limit=10)

    # Reset the environment
    timestep = env.reset()

    # Define a target position for the joints
    target_positions = np.array([-0.4, -0.2, 2.1, 0.1, 1.4, 1.2])

    # Run the simulation
    for step in range(100):
        # Compute the action (move towards the target positions)
        current_positions = timestep.observation['qpos']
        action = (target_positions - start_pos)/100 * step + start_pos
        print("action: ", action)

        # Step the environment
        timestep = env.step(action)

        # Print the current joint positions
        print(f"Step {step}: Joint positions: {timestep.observation['qpos']}")

if __name__ == '__main__':
    test_position_control()