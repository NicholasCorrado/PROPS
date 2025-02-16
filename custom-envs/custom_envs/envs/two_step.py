from typing import Optional

import gymnasium as gym
import numpy as np


class TwoStepEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.action_space = gym.spaces.Discrete(2)
        # self.observation_space = gym.spaces.Discrete(7)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(7,))

        self.step_count = 0
        self.state = 0

        self.dynamics = {
            (0, 0): 1,
            (0, 1): 2,
            (1, 0): 3,
            (1, 1): 4,
            (2, 0): 5,
            (2, 1): 6,
        }

        self.reward_function = {
            0: 0,
            1: 0,
            2: 0,
            3: 2,
            4: 0.5,
            5: 1,
            6: 1,
        }

    def step(self, a):

        self.state = self.dynamics[(self.state, a)]
        reward = self.reward_function[self.state]
        self.step_count += 1

        info = {}
        terminated = False
        truncated = False
        if self.step_count >= 2:
            terminated = True

        state = np.zeros(7)
        state[self.state] = 1
        # print(self.step_count, self.state, reward)
        return state, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        self.step_count = 0
        self.state = 0
        state = np.zeros(7)
        state[self.state] = 1
        return state, {}