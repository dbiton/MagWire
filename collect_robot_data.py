import json
from math import radians
import math
from time import sleep, time
from multiprocessing import Process
from hilbertcurve.hilbertcurve import HilbertCurve
from matplotlib import animation, pyplot as plt
# from move_silly import move_waypoints
from move_silly import get_config, move_waypoints
from robot_controller import BaseRobot
import numpy as np
import os
try:
    import winsound
except ImportError:
    def playsound(frequency, duration):
        os.system(f'speaker-test -t sine -f {frequency}')
else:
    def playsound(frequency,duration):
        winsound.Beep(frequency,duration)

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
            res = {"time": time(), "pos": get_config()}
            sleep(1 / 100)
            json.dump(res, f)

from footage_parser import FootageParser

def save_footage():
    fp = FootageParser()
    fp.parse_video("frames", True, True)

if __name__ == "__main__":
    process = Process(target=log_data)
    process_video = Process(target=save_footage)
    process_video.start()
    process.start()

    # Bottom 4 corners
    v000 = (-700/1000,   0/1000,   -40.0/1000)
    v100 = (-900/1000,   0/1000,   -40.0/1000)
    v010 = (-900/1000,   200/1000,   -40.0/1000)
    v110 = (-700/1000,   200/1000,   -40.0/1000)

    # Top 4 corners
    v001 = (-700/1000,   0/1000,   30/1000)
    v101 = (-900/1000,   0/1000,   30/1000)
    v011 = (-900/1000,   200/1000,   30/1000)
    v111 = (-700/1000,   200/1000,   30/1000)

    corners = [v000, v100, v010, v110, v001, v101, v011, v111]
    
    acceleration = 0.2
    min_velocity = 0.1
    max_velocity = 0.8
    orientation = [0, math.pi, 0]


    steps_velocity = 8
    for order in range(1, 4):
      for velocity in np.linspace(min_velocity, max_velocity, steps_velocity):
        curve_points = space_filling_curve_3D(corners, order=order)
        curve_pts = np.array(curve_points)
        xs, ys, zs = curve_pts[:, 0], curve_pts[:, 1], curve_pts[:, 2]

        waypoints = []
        for curve_point in curve_points:
          waypoint = list(curve_point) + orientation + [velocity, acceleration]
          waypoints.append(waypoint)

        move_waypoints(waypoints)
        
        frequency = 432
        sleep(10)
    process.join() 
