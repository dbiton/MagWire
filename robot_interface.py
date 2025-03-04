

from ur_simulation.building_blocks import Building_Blocks
from ur_simulation.environment import Environment
from ur_simulation.inverse_kinematics import forward_kinematic_solution, DH_matrix_UR5e, inverse_kinematic_solution
from ur_simulation.kinematics import Transform, UR5e_PARAMS
from ur_simulation.planners import RRT_CONNECT
import numpy as np
class RobotInterface:
    def __init__(self):
        self.last_waypoint = None

    def get_current_position(self):
        return self.last_waypoint[:3]

    def estimate_actuator_transform(self, robot_config):
        return forward_kinematic_solution(DH_matrix_UR5e, robot_config)

    def estimate_robot_config(self, actuator_transform):
        return inverse_kinematic_solution(DH_matrix_UR5e, actuator_transform)

    def actuator_pose_to_transform(self, pose: list):
        x, y, z, rx, ry, rz = pose
        Rx = np.array([[1, 0, 0, 0],
                    [0, np.cos(rx), -np.sin(rx), 0],
                    [0, np.sin(rx), np.cos(rx), 0],
                    [0, 0, 0, 1]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry), 0],
                    [0, 1, 0, 0],
                    [-np.sin(ry), 0, np.cos(ry), 0],
                    [0, 0, 0, 1]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0, 0],
                    [np.sin(rz), np.cos(rz), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
        R = Rz @ Ry @ Rx
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    
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
        return path, path_cost, plan_time
            
    def get_config(self):
        raise Exception('Virtual method')
    
    def move_waypoint(self, waypoint):
        raise Exception('Virtual method')
    
    def move_waypoints(self, waypoints: list):
        for waypoint in waypoints:
            self.move_waypoint(waypoint)