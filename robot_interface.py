

import numpy as np
from footage_parser import FootageParser
from move_silly import get_config, move_waypoint
from ur_simulation.building_blocks import Building_Blocks
from ur_simulation.environment import Environment
from ur_simulation.inverse_kinematics import forward_kinematic_solution, DH_matrix_UR5e, inverse_kinematic_solution
from ur_simulation.kinematics import Transform, UR5e_PARAMS
from ur_simulation.planners import RRT_CONNECT
import threading

class RobotInterface:
    def __init__(self):
        self.last_waypoint = None
        self.magwire_pos = [0.5, 0.5]
        self.magwire_last_update = None
        self.fp_thread = threading.Thread(target=self._thread_update_wire_pos)
        self.fp_thread.daemon = True
        self.fp_thread.start()

    def _thread_update_wire_pos(self):
        fp = FootageParser()
        for pos, t in fp.parse_video("frames", True, True):
            self.magwire_pos = pos
            self.magwire_last_update = t

    def get_current_position(self):
        return self.last_waypoint[:3]

    def estimate_actuator_transform(self, robot_config):
        return forward_kinematic_solution(DH_matrix_UR5e, robot_config)

    def estimate_robot_config(self, actuator_transfor):
        return inverse_kinematic_solution(DH_matrix_UR5e, actuator_transfor)

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
        robot_config = get_config()
        return robot_config, self.magwire_pos
    
    def move(self, waypoint):
        move_waypoint(waypoint)
        self.last_waypoint = waypoint