from typing import Optional

import gymnasium as gym
import numpy as np


class Goal2DEnv(gym.Env):
    def __init__(self, delta=0.025, sparse=1, rbf_n=None, d_fourier=None, neural=False, d=1, quadrant=False, center=False, fixed_goal=False):

        self.n = 2
        self.action_space = gym.spaces.Box(low=np.zeros(2), high=np.array([1, 2 * np.pi]), shape=(self.n,))

        self.boundary = 1.05
        self.observation_space = gym.spaces.Box(-self.boundary, +self.boundary, shape=(2 * self.n,), dtype="float64")

        self.step_num = 0
        self.delta = delta

        self.sparse = sparse
        self.d = d
        self.x_norm = None
        self.quadrant = quadrant
        self.center = center
        self.fixed_goal = fixed_goal
        super().__init__()

    def _clip_position(self):
        # Note: clipping makes dynamics nonlinear
        self.x = np.clip(self.x, -self.boundary, +self.boundary)

    def step(self, a):

        self.step_num += 1
        ux = a[0] * np.cos(a[1])
        uy = a[0] * np.sin(a[1])
        u = np.array([ux, uy])

        self.x += u * self.delta
        self._clip_position()

        dist = np.linalg.norm(self.x - self.goal)
        terminated = dist < 0.05
        truncated = False

        if self.sparse:
            reward = +1.0 if terminated else -0.1
        else:
            reward = -dist

        info = {}
        self.obs = np.concatenate((self.x, self.goal))
        return self._get_obs(), reward, terminated, truncated, info

    def _sample_goal(self):
        if self.quadrant:
            goal = np.random.uniform(low=0, high=1, size=(self.n,))
        elif self.fixed_goal:
            goal = np.array([0.5, 0.5])
        else:
            goal = np.random.uniform(low=-self.d, high=self.d, size=(self.n,))
        return goal

    def _get_obs(self):
        return np.concatenate([self.x, self.goal])

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):

        self.step_num = 0

        self.x = np.random.uniform(-1, 1, size=(self.n,))
        self.goal = self._sample_goal()

        dist = np.linalg.norm(self.x - self.goal)
        while dist < 0.05:
            self.x = np.random.uniform(-1, 1, size=(self.n,))
            self.goal =self._sample_goal()
            dist = np.linalg.norm(self.x - self.goal)

        self.obs = np.concatenate((self.x, self.goal))
        return self._get_obs(), {}

class Goal2DQuadrantEnv(Goal2DEnv):
    def __init__(self, d=1, rbf_n=None, d_fourier=None, neural=False):
        super().__init__(delta=0.025, sparse=1, rbf_n=rbf_n, d_fourier=d_fourier, neural=neural, d=d, quadrant=True)