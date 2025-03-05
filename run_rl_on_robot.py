import json
from multiprocessing import Process
from threading import Thread
from time import sleep, time
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from magwire_gym_env import MagwireEnv
from stable_baselines3.common.env_checker import check_env
from robot_controller import RobotController
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
r = RobotController()

def log_data():
    with open("robot_posMar5.txt", "a") as f:
        while True:
            res = {"time": time(), "pos": r.get_config()}
            sleep(1 / 100)
            json.dump(res, f)

def train_rl_model():
    '''
    v000 = [-200/1000,  -550/1000,   310/1000]
    v100 = [-50/1000,    -550/1000,   310/1000]
    v010 = [-50/1000,    -350/1000,   310/1000]
    v110 = [-200/1000,  -350/1000,   310/1000]
    ps = [v000, v100, v010, v110]
    orientation = [0, 3.141, 0]
    velocity, acceleration = 1.0, 0.2
    while True:
        mvs = [p + orientation + [velocity, acceleration] for p in ps]
        r.move_waypoints(mvs)
'''
    process_robot = Thread(target=log_data)
    process_robot.start()

    env = MagwireEnv(r)
    env = Monitor(env)
    check_env(env, warn=True)
    timesteps_per_save = 1024
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_magwire/", n_steps=min(512, timesteps_per_save))
    total_timesteps = 100000
    for i in range(0, total_timesteps, timesteps_per_save):
        print("Iteration", i)
        model.learn(total_timesteps=timesteps_per_save, progress_bar=True)
        model.save("ppo_magwire_model")
        model = PPO.load("ppo_magwire_model")
        model.set_env(env)
    env = MagwireEnv(r)
    obs, info = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, trunc, info = env.step(action)
        if done:
            obs = env.reset()

if __name__=="__main__":
    train_rl_model()