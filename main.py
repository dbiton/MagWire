import json

import pandas as pd
from matplotlib import pyplot as plt
from footage_parser import FootageParser
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from magwire_gym_env import MagwireEnv
from robot_simulator import RobotSimulator


def load_data(filepath: str):
    data = {
        "time": [],
        "robot": [],
        "wire": []
    }
    with open(filepath) as f:
        for line in f:
            v = json.loads(line)
            data["time"].append(v["time"])
            data["robot"].append(v["pos"][0])
            data["wire"].append(v["pos"][1])
    data = pd.DataFrame(data)
    return data

def load_magwire_pos(video_path: str):
    footage_parser = FootageParser()
    wire_pos = [v for v in footage_parser.parse_video(video_path)]
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

def plot_robot_velocity(robot_data):
    ys = [v['pos'] for v in robot_data] 
    x = [v['time'] for v in robot_data]
    ys_sep = [[y[i] for y in ys] for i in range(len(ys[0]))]
    dys_dxs = [np.gradient(y, x) for y in ys_sep]  # Numerical derivative of y with respect to x
    dy_dx = [sum(_dy_dx[i] for _dy_dx in dys_dxs) for i in range(len(dys_dxs[0]))]

    plt.figure()
    plt.plot(x, dy_dx, color='r', label='robot', alpha=0.33)
    vs = [(i, u) for ((i, u), v) in zip(enumerate(x), dy_dx) if abs(v) < 0.01]
    rs = find_ranges(v[0] for v in vs)
    rs = [r for r in rs if r[1]-r[0] > 50]
    vsdict = {k: v for (k,v) in vs}
    stop_intervals = [(vsdict[start], vsdict[end]) for start, end in rs]
    robot_pauses = detect_robot_pauses(robot_data)
    beep_intervals = detect_beeps()
    for start, end in stop_intervals:
        plt.axvspan(start, end, color='b', alpha=0.33)
    for start, end in beep_intervals:
        plt.axvspan(start, end, color='y', alpha=0.33)
    plt.plot(robot_pauses, marker='s', linestyle='-', color='b', label='pauses')
    plt.legend()
    plt.show()

def train_rl_model(env: MagwireEnv):
    check_env(env, warn=True)

    # Wrap your environment for training
    train_env = env

    # Instantiate the PPO model
    model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log="./ppo_magwire/")

    # Train the model
    model.learn(total_timesteps=100000)

    # Save the model
    model.save("ppo_magwire_model.h5")

    # Load and evaluate
    robot_simulator = RobotSimulator(load_model_path="robot_simulator_model.h5")
    env = MagwireEnv(robot_simulator)
    model = PPO.load("ppo_magwire_model.h5")
    obs, info = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, trunc, info = env.step(action)
        if done:
            obs = env.reset()

def main():
    print("loading data...")
    data = load_data('data\\2.03.25\\pos.txt')
    print("training robot simulator...")
    robot_simulator = RobotSimulator(save_model_path="robot_simulator_model.h5")
    robot_simulator.train_model(data)
    print("training RL agent using robot simulator...")
    env = MagwireEnv(robot_simulator)
    train_rl_model(env)
    print("done!")

from scipy.interpolate import interp1d

def create_multidimensional_interpolation_function(points_dict: dict):
    points = [(k,v) for (k,v) in points_dict.items()]
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
    main()