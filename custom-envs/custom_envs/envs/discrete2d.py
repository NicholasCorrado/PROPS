from typing import Optional

import gymnasium as gym
import numpy as np


class Discrete2DEnv(gym.Env):
    def __init__(self, delta=0.05, n=10):

        self.n = n

        self.observation_space = gym.spaces.Box(-1, +1, shape=(4,), dtype="float64")
        self.action_space = gym.spaces.Discrete(self.n)
        self.thetas = np.linspace(0, 2*np.pi, self.n)

        self.delta = delta

        super().__init__()

    def step(self, a):

        theta = self.thetas[a]
        displacement = self.delta * np.array([np.sin(theta), np.cos(theta)])
        self.x += displacement
        self.x = np.clip(self.x, -1, +1)

        dist = np.linalg.norm(self.x - self.goal)
        terminated = dist < 0.05
        truncated = False

        # if self.sparse:
        #     reward = +1.0 if terminated else -0.1
        # else:
        #     reward = -dist
        reward = -dist

        info = {}
        self.obs = np.concatenate((self.x, self.goal))
        return self._get_obs(), reward, terminated, truncated, info

    def _sample_goal(self):
        goal = np.random.uniform(low=-1, high=+1, size=(2,))
        return goal

    def _get_obs(self):
        return np.concatenate([self.x, self.goal])

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):

        self.x = np.random.uniform(-1, 1, size=(2,))
        self.goal = self._sample_goal()

        dist = np.linalg.norm(self.x - self.goal)
        while dist < 0.05:
            self.x = np.random.uniform(-1, 1, size=(2,))
            self.goal =self._sample_goal()
            dist = np.linalg.norm(self.x - self.goal)

        self.obs = np.concatenate((self.x, self.goal))
        return self._get_obs(), {}

