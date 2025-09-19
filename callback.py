from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class RewardLoggerCallback(BaseCallback):
    def __init__(self, log_interval=1000, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.episode_rewards = []
        self.current_rewards = 0

    def _on_step(self) -> bool:
        # Cumulative reward for the current episode
        self.current_rewards += self.locals["rewards"][0]

        # If the episode is done, save the total reward
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_rewards)
            self.current_rewards = 0

            if len(self.episode_rewards) % self.log_interval == 0:
                mean_r = np.mean(self.episode_rewards[-self.log_interval:])
                print(f"Step {self.num_timesteps}: "
                      f"Mean reward over last {self.log_interval} episodes: {mean_r:.3f}")

        return True
