from time import sleep
import urx

robot = urx.Robot("192.168.0.11")
robot.set_tcp((0, 0, 0.1, 0, 0, 0))
robot.set_payload(2, (0, 0, 0.1))

def get_config():
    return robot.getj()

def move_waypoint(waypoint):
    pose = tuple(waypoint[:6])
    velocity = waypoint[6]
    acceleration = waypoint[7]
    robot.movel(pose, vel=velocity, acc=acceleration)
    while robot.is_program_running():
        sleep(0.01)

def move_waypoints(waypoints: list):
    for waypoint in waypoints:
        move_waypoint(waypoint)
