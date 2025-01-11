

import numpy as np
from ur_simulation.building_blocks import Building_Blocks
from ur_simulation.environment import Environment
from ur_simulation.inverse_kinematics import forward_kinematic_solution, DH_matrix_UR5e
from ur_simulation.kinematics import Transform, UR5e_PARAMS
from ur_simulation.planners import RRT_CONNECT

class RobotInterface:
    def __init__(self):
        self.robot_config = np.zeros((6,))
    
    def estimate_actuator_transform(self, robot_config):
        return forward_kinematic_solution(DH_matrix_UR5e, robot_config)
    
    def estimate_robot_path(self, robot_config_initial, robot_config_target):
        ur_params = UR5e_PARAMS(inflation_factor=1)
        env = Environment(env_idx=2)
        transform = Transform(ur_params)
        bb = Building_Blocks(transform=transform, 
                            ur_params=ur_params, 
                            env=env,
                            resolution=0.1, 
                            p_bias=0.05,)
        rrt_star_planner = RRT_CONNECT(max_step_size=0.3,
                                    max_itr=1000,
                                    bb=bb)
        path, path_cost, plan_time = rrt_star_planner.find_path(start_conf=robot_config_initial,
                                                                goal_conf=robot_config_target)
        return path
            
    def get_config(self):
        return self.robot_config
    
    def move(self, robot_config):
        self.robot_config = robot_config