import json
from time import sleep, time
import warnings
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from magwire_gym_env import MagwireEnv
from robot_simulator import RobotSimulator
from stable_baselines3.common.evaluation import evaluate_policy

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Function to create a new instance of the environment with its own simulator
def make_env():
    def _init():
        # Create a new simulator instance for each environment
        r = RobotSimulator(load_model_path="robot_simulator_model.h5")
        env = MagwireEnv(r)
        env = Monitor(env)
        return env
    return _init

def train_rl_model():
    num_envs = 10  # Set the number of parallel environments
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    # Check the first environment for issues
    # check_env(env.envs[0], warn=True)
    
    # Adjust n_steps if necessary (n_steps is per environment; overall batch size becomes num_envs * n_steps)
    n_steps = 512
    timesteps_per_save = n_steps * num_envs
    model = PPO("MlpPolicy", env, verbose=2, tensorboard_log="./ppo_magwire/", n_steps=n_steps)
    total_timesteps = 1000000
    
    for i in range(0, total_timesteps, timesteps_per_save):
        print("Iteration", i)
        model.learn(total_timesteps=timesteps_per_save, progress_bar=True)
        model.save("ppo_magwire_model")
        model = PPO.load("ppo_magwire_model")
        model.set_env(env)
    
    # Demonstration loop: run the trained model in one of the vectorized environments
    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        if any(dones):  # if any of the environments is done, reset them
            obs = env.reset()

if __name__=="__main__":
    env = SubprocVecEnv([make_env() for _ in range(10)])
    model = PPO.load("ppo_magwire_model")
    model.set_env(env)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
    print(f"Evaluation results: Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")
