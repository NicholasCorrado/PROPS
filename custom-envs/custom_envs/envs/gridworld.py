from typing import Optional, Tuple

import gymnasium as gym
import numpy as np


class GridWorldEnv(gym.Env):
    def __init__(self, shape=(5,5), rewards=(-0.01, 0.5, 1)):
        super().__init__()

        self.shape = np.array(shape)
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(np.product(self.shape),))

        self.nrows, self.ncols = self.shape
        self.rowcol = (self.shape-1)/2
        self.init_rowcol = (self.shape-1)/2 # agent starts in middle of the grid.

        self.rewards = rewards[0] * np.ones(shape=self.shape)
        self.rewards[0, 0] = rewards[1] # subopt
        self.rewards[self.nrows-1, self.ncols-1] = rewards[2]
        # self.rewards[:3, :3] = 0.1
        self.opt_reward = rewards[2]

        self.terminals = np.zeros(shape=self.shape, dtype=bool)
        self.terminals[self.rewards > 0] = True
        # self.terminals[self.rewards > 0.02] = True

        print(self.rewards)
        print(self.terminals)


    def _rowcol_to_obs(self, rowcol):
        idx = int(rowcol[0] * self.shape[0] + rowcol[1])
        state = np.zeros(self.observation_space.shape[-1])
        state[idx] = 1
        return state

    def step(self, a):
        # up
        if a == 0:
            self.rowcol[0] -= 1
        # down
        elif a == 1:
            self.rowcol[0] += 1
        # left
        elif a == 2:
            self.rowcol[1] -= 1
        # down
        elif a == 3:
            self.rowcol[1] += 1

        self.rowcol = np.clip(self.rowcol, a_min=np.zeros(2), a_max=self.shape-1).astype(int)

        state = self._rowcol_to_obs(self.rowcol)
        reward = self.rewards[self.rowcol[0], self.rowcol[1]]
        terminated = self.terminals[self.rowcol[0], self.rowcol[1]]
        truncated = False
        info = {'is_success': reward == self.opt_reward}

        return state, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        self.rowcol = self.init_rowcol.copy()
        state = self._rowcol_to_obs(self.rowcol)

        return state, {}


class GridWorld2Env(GridWorldEnv):
    def __init__(self, shape=(5,5), rewards=(-0.01, 0.1, 1)):
        super().__init__(shape, rewards)

        self.shape = np.array(shape)
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(np.product(self.shape),))

        self.nrows, self.ncols = self.shape
        self.rowcol = (self.shape-1)/2
        self.init_rowcol = (self.shape-1)/2 # agent starts in middle of the grid.

        self.rewards = rewards[0] * np.ones(shape=self.shape)
        self.rewards[:3, :3] = 0.01
        self.rewards[2, 2] = rewards[0]
        self.rewards[0, 0] = rewards[1] # subopt
        self.rewards[self.nrows-1, self.ncols-1] = rewards[2]


        self.terminals = np.zeros(shape=self.shape, dtype=bool)
        self.terminals[self.rewards > 0.1] = True

        print(self.rewards)
        print(self.terminals)


class GridWorldContinuingEnv(GridWorldEnv):
    def __init__(self, shape=(5,5)):
        super().__init__()
        self.terminals = np.zeros(shape=self.shape, dtype=bool)
        print(self.rewards)
        print(self.terminals)

class GridWorldCliffEnv(GridWorldEnv):
    def __init__(self, shape=(5,5)):
        super().__init__(shape)
        self.shape = np.array(shape)
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(np.product(self.shape),))

        self.rewards[:, :] = -1
        self.rewards[self.nrows-1, 1:self.ncols-1] = -100 # cliff
        self.rewards[self.nrows-1, self.ncols-1] = 0 # goal

        self.terminals = np.zeros(shape=self.shape, dtype=bool)
        self.terminals[self.nrows-1, 1:] = True # goal state

        print(self.rewards)
        print(self.terminals)
