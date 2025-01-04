import threading
from time import sleep, time
import numpy as np
from numpy import pi
from urx import Robot

class BaseRobot:
    """
    BaseRobot is a class to interface with a UR robot arm.
    It handles connection, movement commands, and specific positions.
    """
    
    ip = '192.168.0.10'
    home = [0, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, -0.005]
    camera_position = [0, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, -0.005]
    movement_delay_interval = 1  # Default sleep time between robot movements

    def __init__(self):
        """
        Initializes the BaseRobot instance and connects to the robot.
        Retries the connection every 5 seconds if it fails.
        """
        self.robot = None
        while self.robot is None:
            try:
                self.robot = Robot(self.ip, use_rt=True)
            except Exception as e:
                print('Cannot connect to robot. Retrying...')
                print(f'Error: {e}')
                sleep(5)

    def move(self, config):
        """
        Moves the robot to a specified joint configuration.
        
        :param config: List of joint angles in radians
        """
        dist = lambda a, b: np.linalg.norm(np.array(a) - np.array(b))
        try:
            self.robot.movej(config, acc=10, vel=0.3)
        except Exception as e:
            print(f'Error moving to config: {e}')
        
        # Wait until the robot reaches the target configuration
        while dist(self.get_config(), config) > 0.1:
            pass

    def move_all(self, config_list):
        """
        Moves the robot through a sequence of joint configurations.
        
        :param config_list: List of joint configurations
        """
        try:
            self.robot.movexs('movej', config_list, acc=50, vel=0.1, radius=0.1)
        except Exception as e:
            print(f'Error moving through config list: {e}')

    def move_home(self):
        """
        Moves the robot to the home position.
        """
        self.move(self.home)

    def move_to_camera_position(self):
        """
        Moves the robot to the camera position.
        """
        self.move(self.camera_position)

    def get_config(self):
        """
        Gets the current joint configuration of the robot.
        
        :return: List of current joint angles in radians
        """
        return self.robot.getj()

    def execute_path(self, path, timing_profile=None):
        """
        Executes a given path with optional timing profile.
        
        :param path: List of joint configurations
        :param timing_profile: List of transition times between each pair of configs
        """
        start_time = time()
        for i, config in enumerate(path):
            self.move(config)
            
            time_to_sleep = self.movement_delay_interval  # Default sleep time
            if timing_profile is not None and time() - start_time < timing_profile[i]:
                time_to_sleep = max(time_to_sleep, timing_profile[i] - (time() - start_time))
            sleep(time_to_sleep)
