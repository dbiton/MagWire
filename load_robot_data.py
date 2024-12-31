import json

from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from footage_parser import parse_video
from detect_beeps import detect_beeps
import numpy as np

from simulator import create_model


def load_robot_pos(filepath: str):
  with open(filepath, "r") as f:
    return json.load(f)

def load_magwire_pos():
    wire_pos = parse_video()
    return wire_pos

def find_ranges(numbers):
    if not numbers:
        return []
    
    # Sort the numbers
    numbers = sorted(numbers)
    
    # Initialize ranges
    ranges = []
    start = numbers[0]
    end = numbers[0]

    for i in range(1, len(numbers)):
        if numbers[i] == numbers[i - 1] + 1:
            # Extend the current range
            end = numbers[i]
        else:
            # Close the current range and start a new one
            ranges.append((start, end))
            start = numbers[i]
            end = numbers[i]

    # Add the last range
    ranges.append((start, end))
    return ranges

def detect_robot_pauses(robot_data):
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

def main():
    robot_data = load_robot_pos('data\\28.11.24\\robot_pos.json')
    start_time = min([v['time'] for v in robot_data]) - 16.936 + 6.4
    for v in robot_data:
        v['time'] -= start_time
    robot_pauses = detect_robot_pauses(robot_data)
    beep_intervals = detect_beeps()

    ys = [v['pos'] for v in robot_data] 
    x = [v['time'] for v in robot_data]
    ys_sep = [[y[i] for y in ys] for i in range(len(ys[0]))]
    dys_dxs = [np.gradient(y, x) for y in ys_sep]  # Numerical derivative of y with respect to x
    dy_dx = [sum(_dy_dx[i] for _dy_dx in dys_dxs) for i in range(len(dys_dxs[0]))]

    plt.figure()

    #plt.plot(x, dy_dx, color='r', label='robot', alpha=0.33)

    vs = [(i, u) for ((i, u), v) in zip(enumerate(x), dy_dx) if abs(v) < 0.01]

    rs = find_ranges(v[0] for v in vs)
    rs = [r for r in rs if r[1]-r[0] > 50]

    vsdict = {k: v for (k,v) in vs}
    stop_intervals = [(vsdict[start], vsdict[end]) for start, end in rs]
    for start, end in stop_intervals:
        plt.axvspan(start, end, color='b', alpha=0.33)
    for start, end in beep_intervals:
        plt.axvspan(start, end, color='y', alpha=0.33)
    
    # plt.plot(robot_pauses, marker='s', linestyle='-', color='b', label='pauses')
    plt.legend()
    #plt.show()

    wire_pos = load_magwire_pos()
    return wire_pos, robot_data

from scipy.interpolate import interp1d

def create_multidimensional_interpolation_function(points):
    points = sorted(points, key=lambda p: p[0])
    x_values = [p[0] for p in points]
    y_values = np.array([p[1] for p in points])  # Convert list of y-lists to a 2D array
    interpolation_funcs = [
        interp1d(x_values, y_values[:, i], kind='linear', fill_value="extrapolate")
        for i in range(y_values.shape[1])
    ]
    def interpolator(x):
        return [func(x) for func in interpolation_funcs]
    return interpolator

if __name__ == "__main__":
    time_step = 0.05
    wire_pos, robot_data = main()
    time_start = max(min([v["time"] for v in wire_pos]), min([v["time"] for v in robot_data]))
    time_end = min(max([v["time"] for v in wire_pos]), max([v["time"] for v in robot_data]))
    wire_pos = [(v["time"], v["pos"]) for v in wire_pos]
    robot_data = [(v["time"], v["pos"]) for v in robot_data]
    wire_pos = create_multidimensional_interpolation_function(wire_pos)
    robot_data = create_multidimensional_interpolation_function(robot_data)
    
    X = []
    y = []
    for t in np.arange(time_start, time_end, time_step):
        print(t)
        X.append(np.concatenate(([time_step], robot_data(t)), axis=0))
        y.append(wire_pos(t))
    X = np.array(X)
    y = np.array(y)
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y)
    create_model(X, y)