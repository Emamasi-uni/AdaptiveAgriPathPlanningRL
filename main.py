from collections import defaultdict
from datetime import datetime
import numpy as np
import os
import time

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
import torch
from tqdm import tqdm
from GaussianFieldEnv import GaussianFieldEnv
from callback import RewardLoggerCallback
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from helper import save_dict


def train_dqn_model(model_path, env_kwargs=None, total_timesteps=100_000, device="cpu"):
    """
    Trains a DQN model on the GaussianFieldEnv environment and saves it to disk.

    Args:
        env_kwargs (dict): Parameters to pass to the environment.
        total_timesteps (int): Number of training steps.
        model_path (str): Path to save the trained model.
    """
    print("Start train")
    train_data = defaultdict(list)
    env_kwargs = env_kwargs or {}
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./models/',
                                             name_prefix=f'dqn_gaussian_{current_datetime}')
    reward_logger = RewardLoggerCallback()  
    callback = CallbackList([reward_logger, checkpoint_callback])

    env = GaussianFieldEnv(**env_kwargs)
    check_env(env)  # Validate the Gym environment

    model = DQN(
        "MlpPolicy",
        env,
        buffer_size=50000,
        device=device,
        verbose=1
    )
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback)

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds.")
    train_data['training_time'].append(training_time)
    train_data["episode_rewards"] = [float(r) for r in reward_logger.episode_rewards]
    train_data["episode_accuracy"] = [float(a) for a in reward_logger.mean_acc_episode]

    save_dict(train_data, f"./data/train_data_{current_datetime}.json")
    model.save(model_path)

    print(f"Model saved to: {model_path}")
    print("Stop train")


def test_dqn_model(model_path, strategy='dqn', env_kwargs=None, n_envs=20, render=False):
    """
    Tests the DQN model on n_envs different environments.

    Args:
        model_path (str): Path to the trained model.
        env_kwargs (dict): Parameters to pass to the environments.
        n_envs (int): Number of test environments.
        max_steps (int): Maximum number of steps per episode.
        render (bool): If True, prints the observation at each step.
    """
    print("Start test")
    test_data = defaultdict(list)

    if strategy != 'random':
        assert os.path.exists(model_path), f"Model not found: {model_path}"
        model = DQN.load(model_path)
        print(f"Model loaded from: {model_path}")
    env_kwargs = env_kwargs or {}

    all_rewards = []
    all_accuracies = []
    all_steps_acc = []

    for i in tqdm(range(n_envs), desc="Testing environments"):
        env = GaussianFieldEnv(**env_kwargs)
        obs = env.reset(seed=i)[0]
        total_reward = 0
        episode_acc = []

        while True:
            if strategy == "random":
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if "cell_accuracy" in info:
                episode_acc.append(info["cell_accuracy"])

            if render:
                print(f"Step Reward: {reward:.4f} | Total: {total_reward:.4f}")
            if terminated or truncated:
                break

        all_rewards.append(total_reward)
        if episode_acc:
            all_accuracies.append(np.mean(episode_acc))
            all_steps_acc.append(episode_acc)

    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    mean_acc = np.mean(all_accuracies) if all_accuracies else None
    std_acc = np.std(all_accuracies) if all_accuracies else None
    print(f"\n Average Reward over {n_envs} envs: {mean_reward:.4f} ± {std_reward:.4f}")
    if mean_acc is not None:
        print(f" Average Accuracy over {n_envs} envs: {mean_acc:.4f} ± {std_acc:.4f}")

    # 🔹 Salvataggio su file
    test_data["all_rewards"] = [float(r) for r in all_rewards]
    test_data["mean_reward"] = mean_reward
    test_data["std_reward"] = std_reward
    test_data["all_accuracies"] = [float(a) for a in all_accuracies]
    test_data["mean_accuracy"] = mean_acc
    test_data["std_accuracy"] = std_acc
    test_data["all_steps_accuracy"] = all_steps_acc

    save_dict(test_data, f"./data/test_data_{current_datetime}.json")

    print("Stop test")


if __name__ == "__main__":
    # Environment parameters
    class grid_info:
        x = 50
        y = 50
        length = 0.125
        shape = (int(y / length), int(x / length))
        center = True

    env_config = {
        "max_steps": 250,
        "grid_info": grid_info,
        "altitudes": 6,
        "fov": 60,
        "binary": True,
        "cluster_radius": 3.0
    }

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    print(f"Using device: {device}")
    
    train_dqn_model(env_kwargs=env_config, total_timesteps=50_000, model_path=f"models/dqn_gaussian_{current_datetime}.zip", device=device)
    test_dqn_model(model_path=f"models/dqn_gaussian_{current_datetime}.zip", strategy='dqn', env_kwargs=env_config, n_envs=20)
