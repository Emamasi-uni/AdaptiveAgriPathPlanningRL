from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.mean_acc_episode = []
        self.step_acc = []
        self.current_rewards = 0
        self.current_acc = []

    def _on_step(self) -> bool:
        self.current_rewards += self.locals["rewards"][0]

        acc = self.locals["infos"][0].get("cell_accuracy")
        if acc is not None:
            self.current_acc.append(acc)

        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_rewards)

            if self.current_acc:
                mean_acc = np.mean(self.current_acc)
                self.mean_acc_episode.append(mean_acc)
                self.step_acc.append(self.current_acc)
                print(f"Episode {len(self.episode_rewards)} finished. "
                      f"Reward={self.current_rewards:.2f}, Accuracy={mean_acc:.3f}")

            self.current_rewards = 0
            self.current_acc = []

        return True
