

import numpy as np
from ur_simulation.inverse_kinematics import forward_kinematic_solution, DH_matrix_UR5e

class RobotInterface:
    def __init__(self):
        self.robot_config = np.zeros((6,))
    
    def get_actuator_transform(self):
        return forward_kinematic_solution(DH_matrix_UR5e, self.robot_config)
    
    def get_config(self):
        return self.robot_config
    
    def move(self, robot_config):
        self.robot_config = robot_config