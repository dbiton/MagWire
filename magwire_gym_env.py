import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from robot_interface import RobotInterface


class MagwireEnv(gym.Env):
    def __init__(self, robot_interface: RobotInterface, max_steps=50):
        super(MagwireEnv, self).__init__()
        
        self.robot_interface = robot_interface
        self.max_steps = max_steps

        # Action space: 6D in [0, 2*pi]
        self.action_space = spaces.Box(
            low=0.0,
            high=math.pi * 2,
            shape=(6,),
            dtype=np.float32
        )

        # Observation space: 2D in [0, 1]
        self.observation_space = spaces.Box(
            low=float('-inf'),
            high=float('inf'),
            shape=(2,),
            dtype=np.float32
        )

        self.steps_taken = None
        self.magwire_curr_pos = None
        self.magwire_target_pos = None

    def reset(self, seed=None, options=None):
        """
        Reset environment state:
         - zero step counter
         - randomly sample a magwire initial position in [0,1]^2
         - randomly sample a target position in [0,1]^2
         - randomly sample a robot configuration
        """
        super().reset(seed=seed)  # Initialize RNG with seed

        self.steps_taken = 0
        
        self.magwire_target_pos = np.random.uniform(low=0.0, high=1.0, size=2).astype(np.float32)
        
        # Random robot config in [0, 2*pi]^6
        request_robot_config = np.random.uniform(low=0.0, high=2 * math.pi, size=6)
        self.robot_interface.move(request_robot_config)
        self.magwire_curr_pos, actual_robot_config = self.robot_interface.get_config()
        
        # Return initial observation and info
        observation = np.array(self.magwire_curr_pos, dtype=np.float32)
        info = {
            'target_pos': self.magwire_target_pos,
            'initial_robot_config': actual_robot_config
        }
        return observation, info

    def step(self, action):
        """
        Apply the chosen action (robot config).
        Then observe the new magwire position from the robot interface.
        Reward = -distance to the target.
        """
        # Clip action to ensure it’s within the valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Move the robot, retrieve the updated magwire position and robot config
        self.robot_interface.move(action)
        self.magwire_curr_pos, robot_config = self.robot_interface.get_config()

        # Distance to target
        dist = np.linalg.norm(self.magwire_curr_pos - self.magwire_target_pos)
        reward = float(-dist)

        # Update step count
        self.steps_taken += 1

        # Check termination conditions
        terminated = bool(dist < 0.01)  # Episode ends if target is reached
        truncated = bool(self.steps_taken >= self.max_steps)  # Episode ends if max steps are reached

        info = {
            'steps_taken': self.steps_taken,
            'robot_config': robot_config,
            'magwire_pos': self.magwire_curr_pos,
            'target_pos': self.magwire_target_pos,
            'distance': dist,
            'reward': reward
        }

        # Observation: magwire's current position (2D)
        observation = np.array(self.magwire_curr_pos, dtype=np.float32)
        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        """
        Print debug info: step count and current distance to target.
        """
        dist = np.linalg.norm(self.magwire_curr_pos - self.magwire_target_pos)
        print(f"Step: {self.steps_taken}, distance to target = {dist:.4f}")

    def close(self):
        """
        Close any resources used by the environment.
        """
        pass
