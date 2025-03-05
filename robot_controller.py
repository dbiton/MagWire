import threading
from time import sleep
import urx
from robot_interface import RobotInterface
from footage_parser import FootageParser

class RobotController(RobotInterface):
    def __init__(self):
        self.robot = urx.Robot("192.168.0.11")
        self.robot.set_tcp((0, 0, 0.1, 0, 0, 0))
        self.robot.set_payload(2, (0, 0, 0.1))
        self.last_waypoint = None
        self.magwire_pos = [0.5, 0.5]
        self.magwire_last_update = None
        self.target = [0.5, 0.5]
        self.fp_thread = threading.Thread(target=self._thread_update_wire_pos)
        self.fp_thread.daemon = True
        self.fp_thread.start()

    def _thread_update_wire_pos(self):
        fp = FootageParser()
        for pos, t in fp.parse_video("frames", True, True, True, self.target):
            self.magwire_pos = pos
            self.magwire_last_update = t
    
    def get_config(self):
        robot_config = self.robot.getj()
        return robot_config, self.magwire_pos

    def move_waypoint(self, waypoint):
        pose = tuple(waypoint[:6])
        velocity = waypoint[6]
        acceleration = waypoint[7]
        self.robot.movel(pose, vel=velocity, acc=acceleration)
        while self.robot.is_program_running():
            sleep(0.01)
        self.last_waypoint = waypoint

    def move_waypoints(self, waypoints: list):
        for waypoint in waypoints:
            self.move_waypoint(waypoint)
