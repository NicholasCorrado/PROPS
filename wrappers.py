"""Set of wrappers for normalizing actions and observations."""
import numpy as np

import gymnasium as gym

class NormalizeObservation(gym.wrappers.NormalizeObservation):
    """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

    Note:
        The normalization depends on past trajectories and observations will not be normalized correctly if the wrapper was
        newly instantiated or the policy was changed recently.
    """

    def __init__(self, env: gym.Env, epsilon: float = 1e-8):
        """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

        Args:
            env (Env): The environment to apply the wrapper
            epsilon: A stability parameter that is used when scaling the observations.
        """
        super().__init__(env, epsilon)
        self.train = True

    def set_train(self):
        self.train = True

    def set_eval(self):
        self.train = False

    def step(self, action):
        """Steps through the environment and normalizes the observation."""
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        if self.train:
            if self.is_vector_env:
                obs = self.normalize(obs)
            else:
                obs = self.normalize(np.array([obs]))[0]
        return obs, rews, terminateds, truncateds, infos

    def reset(self, **kwargs):
        """Resets the environment and normalizes the observation."""
        obs, info = self.env.reset(**kwargs)

        if self.train:
            if self.is_vector_env:
                return self.normalize(obs), info
            else:
                return self.normalize(np.array([obs]))[0], info
        else:
            return obs, info

    def normalize(self, obs):
        """Normalises the observation using the running mean and variance of the observations."""
        if self.train:
            self.obs_rms.update(obs)
        return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)

    def unnormalize(self, obs):
        return obs * np.sqrt(self.obs_rms.var + self.epsilon) + self.obs_rms.mean

class NormalizeReward(gym.wrappers.NormalizeReward):
    r"""This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.

    The exponential moving average will have variance :math:`(1 - \gamma)^2`.

    Note:
        The scaling depends on past trajectories and rewards will not be scaled correctly if the wrapper was newly
        instantiated or the policy was changed recently.
    """

    def __init__(self, env: gym.Env, gamma: float = 0.99, epsilon: float = 1e-8):
        """This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.

        Args:
            env (env): The environment to apply the wrapper
            epsilon (float): A stability parameter
            gamma (float): The discount factor that is used in the exponential moving average.
        """
        super().__init__(env, gamma, epsilon)
        self.train = True

    def set_train(self):
        self.train = True

    def set_eval(self):
        self.train = False

    def step(self, action):
        """Steps through the environment, normalizing the rewards returned."""
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        if not self.is_vector_env:
            rews = np.array([rews])
        self.returns = self.returns * self.gamma * (1 - terminateds) + rews
        if self.train:
            rews = self.normalize(rews)
        if not self.is_vector_env:
            rews = rews[0]
        return obs, rews, terminateds, truncateds, infos

    def normalize(self, rews):
        """Normalizes the rewards with the running mean rewards and their variance."""
        if self.train:
            self.return_rms.update(self.returns)
        return rews / np.sqrt(self.return_rms.var + self.epsilon)

    def unnormalize(self, rews: np.ndarray):
        return rews * np.sqrt(self.return_rms.var + self.epsilon)
