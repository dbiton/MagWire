import math
import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from robot_interface import RobotInterface


def create_box(vertices):
    # Extract the x, y, z coordinates
    x_coords = [v[0] for v in vertices]
    y_coords = [v[1] for v in vertices]
    z_coords = [v[2] for v in vertices]
    
    # Calculate the min and max values for x, y, and z
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)
    z_min = min(z_coords)
    z_max = max(z_coords)
    
    # Return the min and max values that define the box
    return (x_min, x_max, y_min, y_max, z_min, z_max)

def is_point_inside_box(point, box):
    # Point is a tuple (x, y, z)
    x, y, z = point
    
    # Box is a tuple (x_min, x_max, y_min, y_max, z_min, z_max)
    x_min, x_max, y_min, y_max, z_min, z_max = box
    
    # Check if the point is inside the box
    if x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max:
        return True
    else:
        return False

class MagwireEnv(gym.Env):
    def __init__(self, robot_interface: RobotInterface, max_steps=25):
        super(MagwireEnv, self).__init__()
        
        self.robot_interface = robot_interface
        self.max_steps = max_steps
        self.radius = 30/1000
        self.velocity = 2
        self.acceleration = 1.0
    
        # Bottom 4 corners
        v000 = [-200/1000,  -550/1000,   310/1000]
        v100 = [-50/1000,    -550/1000,   310/1000]
        v010 = [-50/1000,    -350/1000,   310/1000]
        v110 = [-200/1000,  -350/1000,   310/1000]

        # Top 4 corners
        v001 = [-200/1000,  -550/1000,   310/1000]
        v101 = [-50/1000,    -550/1000,   310/1000]
        v011 = [-50/1000,    -350/1000,   310/1000]
        v111 = [-200/1000,  -350/1000,   310/1000]

        corners = [v000, v100, v010, v110, v001, v101, v011, v111]

        self.actuator_bbox = create_box(corners)

        self.action_space = gym.spaces.Discrete(5)
        
        magwire_position_space = gym.spaces.Box(low=np.array([-2.0, -2.0]), high=np.array([2.0, 2.0]), dtype=np.float64)
        robot_config_space = gym.spaces.Box(low=np.array([-np.pi] * 6), high=np.array([np.pi] * 6), dtype=np.float64)
        actuator_bbox_mins = [self.actuator_bbox[i] for i in [0,2,4]]
        actuator_bbox_maxs = [self.actuator_bbox[i] for i in [1,3,5]]
        actuator_pos_space = gym.spaces.Box(low=np.array(actuator_bbox_mins), high=np.array(actuator_bbox_maxs), dtype=np.float64)
        
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([magwire_position_space.low, robot_config_space.low, actuator_pos_space.low]),
            high=np.concatenate([magwire_position_space.high, robot_config_space.high, actuator_pos_space.high]),
            dtype=np.float64
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
        print("TARGET:", self.magwire_target_pos)
        self.magwire_target_pos = np.random.uniform(low=0.0, high=1.0, size=2).astype(np.float64)
        self.robot_interface.target[0] = self.magwire_target_pos[0]
        self.robot_interface.target[1] = self.magwire_target_pos[1]

        orientation = [0, math.pi, 0]
        position = [
            random.uniform(self.actuator_bbox[0], self.actuator_bbox[1]),
            random.uniform(self.actuator_bbox[2], self.actuator_bbox[3]),
            random.uniform(self.actuator_bbox[4], self.actuator_bbox[5]),
        ]
        movement = position + orientation + [self.velocity, self.acceleration]
        self.robot_interface.move_waypoint(movement)
        magwire_pos, robot_config = self.robot_interface.get_config()
        actuator_pos = self.robot_interface.get_current_position()
        observation = np.concatenate([magwire_pos, robot_config, actuator_pos])
        info = {
            'target_pos': self.magwire_target_pos,
        }
        return observation, info

    def step(self, action):
        """
        Apply the chosen action (robot config).
        Then observe the new magwire position from the robot interface.
        Reward = -distance to the target.
        """
        
        directions = [
            [ 0, 0, 0],
            [ 0, 0, 1],
            [ 0, 0,-1],
            [ 0, 1, 0],
            [ 0,-1, 0],
            #[ 1, 0, 0], # DISABLE UP /
            #[-1, 0, 0], #
        ]

        direction = directions[action]
        position = list(self.robot_interface.get_current_position() + self.radius * np.array(direction))
        
        if not is_point_inside_box(position, self.actuator_bbox):
            position = self.robot_interface.get_current_position()

        orientation = [0, math.pi, 0]
        movement = position + orientation + [self.velocity, self.acceleration]
        self.robot_interface.move_waypoint(movement)
        robot_config, magwire_pos = self.robot_interface.get_config()
        actuator_pos = self.robot_interface.get_current_position()
        observation = np.concatenate([magwire_pos, robot_config, actuator_pos])

        # Distance to target
        dist = np.linalg.norm(magwire_pos - self.magwire_target_pos)
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
