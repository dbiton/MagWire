import json

from matplotlib import pyplot as plt
from footage_parser import parse_video
from detect_beeps import detect_beeps
import numpy as np

def load_robot_pos(filepath: str):
  with open(filepath, "r") as f:
    return json.load(f)

def load_magwire_pos():
    wire_pos = parse_video()
    return wire_pos

def find_robot_pauses(robot_data):
    flat_ranges = []
    current_start = None

    y_values = [sum(v['pos']) for v in robot_data] 
    x_values = [v['time'] for v in robot_data]
    
    for i in range(len(y_values) - 1):
        if y_values[i + 1] - y_values[i] == 0:
            if current_start is None:
                current_start = x_values[i]
        else:
            if current_start is not None:
                flat_ranges.append((current_start, x_values[i]))
                current_start = None

    # Add the last range if the data ends with a flat segment
    if current_start is not None:
        flat_ranges.append((current_start, x_values[-1]))

    flat_ranges = [r for r in flat_ranges if r[1]-r[0] > 0.9]
    return flat_ranges

robot_data = load_robot_pos('data\\28.11.24\\robot_pos.json')
robot_pauses = find_robot_pauses(robot_data)
beeps = detect_beeps()

y = [sum(v['pos']) for v in robot_data] 
x = [v['time'] for v in robot_data]
dy_dx = np.gradient(y, x)  # Numerical derivative of y with respect to x

plt.figure()

plt.plot(x, dy_dx, color='r', label='robot')
# plt.plot(robot_pauses, marker='s', linestyle='-', color='b', label='pauses')
plt.legend()

plt.show()

# wire_pos = load_magwire_pos()
# x = 3