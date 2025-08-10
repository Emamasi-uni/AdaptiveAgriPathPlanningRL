import numpy as np
import os

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from tqdm import tqdm
from GaussianFieldEnv import GaussianFieldEnv


def train_dqn_model(env_kwargs=None, total_timesteps=100_000, model_path="dqn_model.zip"):
    """
    Trains a DQN model on the GaussianFieldEnv environment and saves it to disk.

    Args:
        env_kwargs (dict): Parameters to pass to the environment.
        total_timesteps (int): Number of training steps.
        model_path (str): Path to save the trained model.
    """
    env_kwargs = env_kwargs or {}

    env = GaussianFieldEnv(**env_kwargs)
    check_env(env)  # Validate the Gym environment

    model = DQN(
        "MlpPolicy",
        env,
        buffer_size=50000,
        verbose=1
    )
    model.learn(total_timesteps=total_timesteps)

    model.save(model_path)
    print(f"Model saved to: {model_path}")


def test_dqn_model(model_path="dqn_model.zip", env_kwargs=None, n_envs=20, max_steps=200, render=False):
    """
    Tests the DQN model on n_envs different environments.

    Args:
        model_path (str): Path to the trained model.
        env_kwargs (dict): Parameters to pass to the environments.
        n_envs (int): Number of test environments.
        max_steps (int): Maximum number of steps per episode.
        render (bool): If True, prints the observation at each step.
    """
    assert os.path.exists(model_path), f"Model not found: {model_path}"

    model = DQN.load(model_path)
    env_kwargs = env_kwargs or {}

    all_rewards = []

    for i in tqdm(range(n_envs), desc="Testing environments"):
        env = GaussianFieldEnv(**env_kwargs)
        obs = env.reset(seed=i)[0]
        total_reward = 0

        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if render:
                print(f"Step Reward: {reward:.4f} | Total: {total_reward:.4f}")
            if terminated:
                break

        all_rewards.append(total_reward)

    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    print(f"\n Average Reward over {n_envs} envs: {mean_reward:.4f} ± {std_reward:.4f}")


if __name__ == "__main__":
    # Environment parameters
    class grid_info:
        x = 50
        y = 50
        length = 0.125
        shape = (int(y / length), int(x / length))
        center = True

    env_config = {
        "grid_info": grid_info,
        "altitudes": 6,
        "fov": 60,
        "binary": True,
        "cluster_radius": 3.0
    }

    train_dqn_model(env_kwargs=env_config, total_timesteps=100_000, model_path="dqn_gaussian.zip")
    test_dqn_model(model_path="dqn_gaussian.zip", env_kwargs=env_config, n_envs=20)
