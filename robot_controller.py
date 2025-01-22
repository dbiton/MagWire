import threading
from time import sleep, time
import numpy as np
from numpy import pi
from ur_simulation.inverse_kinematics import forward_kinematic_solution, DH_matrix_UR5e, inverse_kinematic_solution
from urx import Robot

class BaseRobot:
    """
    BaseRobot is a class to interface with a UR robot arm.
    It handles connection, movement commands, and specific positions.
    """
    
    ip = '192.168.0.11'
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
                self.robot = Robot(self.ip)
                sleep(2)
            except Exception as e:
                print('Cannot connect to robot. Retrying...')
                print(f'Error: {e}')
                sleep(5)

    def move(self, config, acceleration=10, velocity=0.3):
        """
        Moves the robot to a specified joint configuration.
        
        :param config: List of joint angles in radians
        """
        dist = lambda a, b: np.linalg.norm(np.array(a) - np.array(b))
        self.robot.movel(config, acceleration, velocity)
    
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
    
import numpy as np
from math import cos, sin

def get_goal_configuration(x, y, z, alpha=-np.pi, beta=0, gamma=0):
    """
    Calculates the robot's joint configuration that achieves the desired position (x, y, z)
    and orientation (defined by alpha, beta, gamma) in Cartesian space.

    :param x, y, z: Target position in Cartesian space.
    :param alpha, beta, gamma: Euler angles for the desired orientation.
    :return: A list of joint angles (6 DOF) or None if no solution is found.
    """

    # Translation values
    tx = x
    ty = y
    tz = z

    # Transformation matrix calculation (position and orientation)
    transform = np.matrix([
        [cos(beta) * cos(gamma), sin(alpha) * sin(beta) * cos(gamma) - cos(alpha) * sin(gamma),
         cos(alpha) * sin(beta) * cos(gamma) + sin(alpha) * sin(gamma), tx],
        [cos(beta) * sin(gamma), sin(alpha) * sin(beta) * sin(gamma) + cos(alpha) * cos(gamma),
         cos(alpha) * sin(beta) * sin(gamma) - sin(alpha) * cos(gamma), ty],
        [-sin(beta), sin(alpha) * cos(beta), cos(alpha) * cos(beta), tz],
        [0, 0, 0, 1]
    ])

    # Calculate inverse kinematics solutions based on the transformation matrix
    IKS = inverse_kinematic_solution(DH_matrix_UR5e, transform)

    # Prepare candidate solutions
    candidate_sols = [IKS[:, i] for i in range(IKS.shape[1])]

    sols = []
    for candidate_sol in candidate_sols:
        # Angle wrapping to ensure joint limits
        for idx, angle in enumerate(candidate_sol):
            if 2 * np.pi > angle > np.pi:
                candidate_sol[idx] = -(2 * np.pi - angle)
            elif -2 * np.pi < angle < -np.pi:
                candidate_sol[idx] = -(2 * np.pi + angle)

        if np.max(candidate_sol) > np.pi or np.min(candidate_sol) < -np.pi:
            continue  # Skip invalid solutions

        sols.append(candidate_sol)

    # Verify solutions by checking the difference between the desired and computed position
    final_sol = []
    for sol in sols:
        transform_check = forward_kinematic_solution(DH_matrix_UR5e, sol)
        diff = np.linalg.norm(np.array([transform_check[0, 3], transform_check[1, 3], transform_check[2, 3]]) -
                              np.array([tx, ty, tz]))

        if diff < 0.05:
            final_sol.append(sol)

    final_sol = np.array(final_sol)

    if len(final_sol) == 0:
        print("No solutions found.")
        return None

    # Return the first valid solution's joint angles (first 6 DOF)
    final_sol = final_sol.tolist()[0][:6]
    return [value for sublist in final_sol for value in sublist]  # Flatten the list if needed
    
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
