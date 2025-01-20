import json
from math import radians
import math
from time import sleep, time
from multiprocessing import Process
import winsound
from hilbertcurve.hilbertcurve import HilbertCurve
from matplotlib import animation, pyplot as plt
from robot_controller import BaseRobot
import numpy as np


def hilbert_curve_3D(order):
    dimension = 3
    hilb = HilbertCurve(p=order, n=dimension)
    side_length = 2**order
    curve_points = []
    for dist in range(side_length**dimension):
        coords = hilb.point_from_distance(dist)
        x = coords[0] / (side_length - 1)
        y = coords[1] / (side_length - 1)
        z = coords[2] / (side_length - 1)
        curve_points.append((x, y, z))
    return curve_points


def trilinear_map_8pts(pt, corners):
    x, y, z = pt
    v000 = np.array(corners[0])
    v100 = np.array(corners[1])
    v010 = np.array(corners[2])
    v110 = np.array(corners[3])
    v001 = np.array(corners[4])
    v101 = np.array(corners[5])
    v011 = np.array(corners[6])
    v111 = np.array(corners[7])
    mapped = (
        (1 - x) * (1 - y) * (1 - z) * v000
        + x * (1 - y) * (1 - z) * v100
        + (1 - x) * y * (1 - z) * v010
        + x * y * (1 - z) * v110
        + (1 - x) * (1 - y) * z * v001
        + x * (1 - y) * z * v101
        + (1 - x) * y * z * v011
        + x * y * z * v111
    )
    return mapped

def space_filling_curve_3D(corners, order):
    hilbert_points = hilbert_curve_3D(order)
    mapped_points = []
    for pt in hilbert_points:
        mapped_pt = trilinear_map_8pts(pt, corners)
        mapped_points.append(tuple(mapped_pt))  # convert np.array -> tuple
    return mapped_points

def to_rad(p):
    return [radians(a) for a in p]

def log_data():
    with open("robot_pos.txt", "w") as f:
        while True:
            res = {"time": time(), "pos": r.get_config()}
            sleep(1 / 100)
            json.dump(res, f)

if __name__ == "__main__":
    robot = BaseRobot()
  
    process = Process(target=log_data)
    process.start()

    # Bottom 4 corners
    v000 = (0.0,   0.0,   0.0)
    v100 = (2.0,   0.3,  -0.5)
    v010 = (0.2,   3.0,   0.1)
    v110 = (2.2,   3.3,  -0.4)

    # Top 4 corners
    v001 = (0.4,   0.1,   2.2)
    v101 = (2.4,   0.35,  1.7)
    v011 = (0.7,   3.1,   2.3)
    v111 = (2.6,   3.35,  1.7)

    corners = [v000, v100, v010, v110, v001, v101, v011, v111]
    
    acceleration = 1
    min_velocity = 0.1
    max_velocity = 10
    orientation = [0, math.pi, 0]
    steps_velocity = 4
    for velocity in np.linspace(min_velocity, max_velocity, steps_velocity):
      for order in range(1, 4):
        curve_points = space_filling_curve_3D(corners, order=order)
        curve_pts = np.array(curve_points)
        xs, ys, zs = curve_pts[:, 0], curve_pts[:, 1], curve_pts[:, 2]

        waypoints = []
        for curve_point in curve_points:
          waypoint = list(curve_point) + orientation + [acceleration, velocity]
          waypoints.append(waypoint)

        robot.move_waypoints(waypoints)
        
        frequency = 432
        duration = 10000
        winsound.Beep(int(frequency), duration)
    process.join()
