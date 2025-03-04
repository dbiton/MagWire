from stable_baselines3 import PPO
from magwire_gym_env import MagwireEnv
from stable_baselines3.common.env_checker import check_env

from robot_interface import RobotInterface


def train_rl_model():
    robot_interface = RobotInterface()
    env = MagwireEnv(robot_interface)
    check_env(env, warn=True)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_magwire/")
    model.learn(total_timesteps=100000)
    # model.save("ppo_magwire_model.h5")
    env = MagwireEnv(robot_interface)
    model = PPO.load("ppo_magwire_model.h5")
    obs, info = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, trunc, info = env.step(action)
        if done:
            obs = env.reset()

if __name__=="__main__":
    train_rl_model()