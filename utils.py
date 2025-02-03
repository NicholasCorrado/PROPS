import argparse
import copy
import glob
import os
import pickle

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import yaml
# from gymnasium.
from torch.distributions import Normal, Categorical

class ConfigLoader(yaml.SafeLoader):

    def __init__(self, stream):
        super().__init__(stream)
        self.add_constructor('!python/object:argparse.Namespace', construct_argpase_namespace)
        self.add_constructor('!python/object/apply:torch.device', construct_device)

def construct_device(device):
    return torch.device(device)

def construct_argpase_namespace(**kwargs):
    argparse.Namespace(**kwargs)

def get_latest_run_id(save_dir: str) -> int:
    max_run_id = 0
    for path in glob.glob(os.path.join(save_dir, 'run_[0-9]*')):
        filename = os.path.basename(path)
        ext = filename.split('_')[-1]
        if ext.isdigit() and int(ext) > max_run_id:
            max_run_id = int(ext)
    return max_run_id


class StoreDict(argparse.Action):
    """
    Custom argparse action for storing dict.

    In: args1:0.0 args2:"dict(a=1)"
    Out: {'args1': 0.0, arg2: dict(a=1)}
    """

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDict, self).__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        arg_dict = {}
        for arguments in values:
            key = arguments.split(":")[0]
            value = ":".join(arguments.split(":")[1:])
            # Evaluate the string as python code
            arg_dict[key] = eval(value)
        setattr(namespace, self.dest, arg_dict)



class NormalizeObservation(gym.Wrapper):
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
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        if self.is_vector_env:
            self.obs_rms = RunningMeanStd(shape=self.single_observation_space.shape)
        else:
            self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.epsilon = epsilon
        self.do_update = True

    def set_update(self, do_update):
        self.do_update = do_update

    def step(self, action):
        """Steps through the environment and normalizes the observation."""
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        if self.is_vector_env:
            obs = self.normalize(obs)
        else:
            obs = self.normalize(np.array([obs]))[0]
        return obs, rews, terminateds, truncateds, infos

    def reset(self, **kwargs):
        """Resets the environment and normalizes the observation."""
        obs, info = self.env.reset(**kwargs)

        if self.is_vector_env:
            return self.normalize(obs), info
        else:
            return self.normalize(np.array([obs]))[0], info

    def normalize(self, obs):
        """Normalises the observation using the running mean and variance of the observations."""
        if self.do_update:
            self.obs_rms.update(obs)
        return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)

    def unnormalize(self, normalized_obs):
        return normalized_obs.cpu() * np.sqrt(self.obs_rms.var + self.epsilon) + self.obs_rms.mean

class NormalizeReward(gym.core.Wrapper):
    r"""This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.

    The exponential moving average will have variance :math:`(1 - \gamma)^2`.

    Note:
        The scaling depends on past trajectories and rewards will not be scaled correctly if the wrapper was newly
        instantiated or the policy was changed recently.
    """

    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ):
        """This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.

        Args:
            env (env): The environment to apply the wrapper
            epsilon (float): A stability parameter
            gamma (float): The discount factor that is used in the exponential moving average.
        """
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        self.return_rms = RunningMeanStd(shape=())
        self.returns = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon
        self.do_update = True

    def set_update(self, do_update):
        self.do_update = do_update

    def step(self, action):
        """Steps through the environment, normalizing the rewards returned."""
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        if not self.is_vector_env:
            rews = np.array([rews])
        self.returns = self.returns * self.gamma * (1 - terminateds) + rews
        if self.do_update:
            rews = self.normalize(rews)
        if not self.is_vector_env:
            rews = rews[0]
        return obs, rews, terminateds, truncateds, infos

    def normalize(self, rews):
        """Normalizes the rewards with the running mean rewards and their variance."""
        if self.do_update:
            self.return_rms.update(self.returns)
        return rews / np.sqrt(self.return_rms.var + self.epsilon)

    def unnormalize(self, normalized_rews):
        """Normalizes the rewards with the running mean rewards and their variance."""
        return normalized_rews * np.sqrt(self.return_rms.var + self.epsilon)


def normalize_obs(obs_rms, obs):
    """Normalises the observation using the running mean and variance of the observations."""
    return torch.Tensor((obs.cpu().numpy() - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8))

def normalize_reward(return_rms, rewards):
    """Normalizes the rewards with the running mean rewards and their variance."""
    return rewards / np.sqrt(return_rms.var + 1e-8)

def make_env(env_id, idx, capture_video, run_name, gamma, device, env_kwargs):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array", **env_kwargs)
        else:
            env = gym.make(env_id, **env_kwargs)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.ClipAction(env)
        env = NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, relu=False):

        if relu:
            activation_fn = nn.ReLU
        else:
            activation_fn = nn.Tanh

        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            activation_fn(),
            layer_init(nn.Linear(64, 64)),
            activation_fn(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            activation_fn(),
            layer_init(nn.Linear(64, 64)),
            activation_fn(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
            nn.Tanh(),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, action_mean, action_std, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x),

    def get_action(self, x, noise=False):
        action_mean = self.actor_mean(x)
        if noise:
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            probs = Normal(action_mean, action_std)
            action = probs.sample()
        else:
            action = action_mean
        return action

    def get_action_and_info(self, x, action=None, clamp=False):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        if clamp:
            action_logstd = torch.clamp(action_logstd, min=-3, max=1)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, action_mean, action_std, probs.log_prob(action).sum(1), probs.entropy().sum(1)

    def sample_actions(self, x):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        action = probs.sample()
        return action

    def sample_actions_unif(self, x):
        with torch.no_grad():
            action_mean = self.actor_mean(x)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            probs = torch.distributions.Uniform(low=torch.clamp(action_mean-3*action_std,-1,+1), high=torch.clamp(action_mean+3*action_std,-1,+1))
            action = probs.sample()
        return action

class AgentDiscrete(nn.Module):
    def __init__(self, envs, relu=False, linear=False):

        if relu:
            activation_fn = nn.ReLU
        else:
            activation_fn = nn.Tanh

        super().__init__()
        if isinstance(envs.single_observation_space, gym.spaces.Box):
            input_dim = np.array(envs.single_observation_space.shape).prod()
        else:
            input_dim = np.array(envs.single_observation_space.n)

        self.obs_dim = input_dim

        self.critic = nn.Sequential(
            layer_init(nn.Linear(input_dim, 64)),
            activation_fn(),
            layer_init(nn.Linear(64, 64)),
            activation_fn(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(input_dim, 64)),
            activation_fn(),
            layer_init(nn.Linear(64, 64)),
            activation_fn(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )


        if linear:
            self.critic = nn.Sequential(
                layer_init(nn.Linear(input_dim, 1), std=0),
                # activation_fn(),
                # layer_init(nn.Linear(64, 64)),
                # activation_fn(),
                # layer_init(nn.Linear(64, 1), std=1.0),
            )
            self.actor = nn.Sequential(
                layer_init(nn.Linear(input_dim, envs.single_action_space.n), std=0),
                # activation_fn(),
                # layer_init(nn.Linear(64, 64)),
                # activation_fn(),
                # layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
            )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.log_prob(action), probs.entropy(), self.critic(x),

    def get_action(self, x, noise=False):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if noise:
            action = probs.sample()
        else:
            action = probs.sample() # @TODO
        return action

    def get_action_and_info(self, x, action=None, clamp=False):
        logits = self.actor(x)
        if clamp:
            logits = torch.clamp(logits, min=-3, max=1)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.log_prob(action), probs.entropy()

    def sample_actions(self, x):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action

    def get_pi(self):
        pi = []
        with torch.no_grad():
            x = torch.zeros(self.obs_dim)
            for i in range(self.obs_dim):
                x[:] = 0
                x[i] = 1
                logits = self.actor(x)
                probs = Categorical(logits=logits).probs.detach().numpy()
                pi.append(probs)
        return np.array(pi)

    def get_pi_s(self, x):
        with torch.no_grad():
            logits = self.actor(x)
            probs = Categorical(logits=logits).probs.detach().numpy()
        return probs


class Evaluate:
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param callback_after_eval: Callback to trigger after every evaluation
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about evaluation results
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
            self,
            model,
            eval_env,
            n_eval_episodes: int = 5,
            eval_freq: int = 10000,
            log_path: str = None,
            suffix: str = '',
            save_model: bool = False,
            deterministic: bool = True,
            device=None,
    ):
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.device = device

        self.save_model = save_model
        self.model = model
        self.eval_env = eval_env
        self.best_model_save_path = log_path
        self.suffix = suffix

        # Logs will be written in ``evaluations.npz``
        os.makedirs(name=log_path, exist_ok=True)
        if log_path is not None:
            if self.suffix != '':
                self.log_path = os.path.join(log_path, f"evaluations_{suffix}")
            else:
                self.log_path = os.path.join(log_path, f"evaluations")
        self.evaluations_returns = []
        self.evaluations_timesteps = []
        self.evaluations_successes = []
        # For computing success rate
        self._is_success_buffer = []

    def evaluate(self, t, train_env, noise):
        # if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:

        env_reward_normalize = train_env.envs[0].env
        env_obs_normalize = train_env.envs[0].env.env.env

        self.eval_env = copy.deepcopy(train_env)
        self.eval_env.envs[0].set_update(False)
        returns, successes = self._evaluate(noise=noise)
        self.eval_env.envs[0].set_update(True)

        if self.log_path is not None:
            self.evaluations_timesteps.append(t)
            self.evaluations_returns.append(returns)
            self.evaluations_successes.append(successes)

            np.savez(
                self.log_path,
                timesteps=self.evaluations_timesteps,
                returns=self.evaluations_returns,
                successes=self.evaluations_successes,
            )

            mean_reward, std_reward = np.mean(returns), np.std(returns)
            mean_success, std_success = np.mean(successes), np.std(successes)

            self.last_mean_reward = mean_reward

            print(f"Eval num_timesteps={t}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
            print(f"Eval num_timesteps={t}, " f"episode_success={mean_success:.2f} +/- {std_success:.2f}")

            # if mean_reward > self.best_mean_reward:
            #     print("New best mean reward!")
            if self.save_model:
                torch.save(self.model, os.path.join(self.best_model_save_path, "best_model.zip"))
                with open(f'{self.best_model_save_path}/env_obs_normalize', 'wb') as f:
                    pickle.dump(env_obs_normalize.obs_rms, f)
                with open(f'{self.best_model_save_path}/env_reward_normalize', 'wb') as f:
                    pickle.dump(env_reward_normalize.return_rms, f)

            # self.best_mean_reward = mean_reward

        return mean_reward, std_reward

    def _evaluate(self, noise):
        eval_returns = []
        eval_successes = []

        obs, _ = self.eval_env.reset()
        for episode_i in range(self.n_eval_episodes):
            ep_returns = []
            ep_successes = []
            done = False
            step = 0
            while not done:
                step += 1
                # ALGO LOGIC: put action logic here
                with torch.no_grad():
                    actions = self.model.get_action(torch.Tensor(obs).to(self.device), noise=noise)
                    # actions = self.model(torch.Tensor(obs).to(self.device))
                    actions = actions.cpu().numpy().clip(self.eval_env.single_action_space.low,
                                                         self.eval_env.single_action_space.high)

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, rewards, terminateds, truncateds, infos = self.eval_env.step(actions)
                done = terminateds[0] or truncateds[0]

                # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
                obs = next_obs

                ep_returns.append(rewards[0])
                ep_successes.append(terminateds[0])

            eval_returns.append(np.sum(ep_returns))
            eval_successes.append(np.sum(ep_successes) * 100)

        return eval_returns, eval_successes



class EvaluateDiscrete:
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param callback_after_eval: Callback to trigger after every evaluation
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about evaluation results
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
            self,
            model,
            eval_env,
            n_eval_episodes: int = 5,
            eval_freq: int = 10000,
            log_path: str = None,
            suffix: str = '',
            save_model: bool = False,
            deterministic: bool = True,
            device=None,
    ):
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.device = device

        self.save_model = save_model
        self.model = model
        self.eval_env = eval_env
        self.best_model_save_path = log_path
        self.suffix = suffix

        # Logs will be written in ``evaluations.npz``
        os.makedirs(name=log_path, exist_ok=True)
        if log_path is not None:
            if self.suffix != '':
                self.log_path = os.path.join(log_path, f"evaluations_{suffix}")
            else:
                self.log_path = os.path.join(log_path, f"evaluations")
        self.evaluations_returns = []
        self.evaluations_timesteps = []
        self.evaluations_successes = []
        # For computing success rate
        self._is_success_buffer = []

    def evaluate(self, t, train_env, noise, pi=None):
        # if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:

        self.eval_env = copy.deepcopy(train_env)
        returns, successes, sa_counts = self._evaluate(noise=noise, pi=pi)

        if self.log_path is not None:
            self.evaluations_timesteps.append(t)
            self.evaluations_returns.append(returns)
            self.evaluations_successes.append(successes)

            np.savez(
                self.log_path,
                timesteps=self.evaluations_timesteps,
                returns=self.evaluations_returns,
                successes=self.evaluations_successes,
            )

            mean_reward, std_reward = np.mean(returns), np.std(returns)
            mean_success, std_success = np.mean(successes), np.std(successes)

            self.last_mean_reward = mean_reward

            print(f"Eval num_timesteps={t}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
            print(f"Eval num_timesteps={t}, " f"episode_success={mean_success:.2f} +/- {std_success:.2f}")

            if mean_reward > self.best_mean_reward:
                print("New best mean reward!")
                if self.save_model:
                    torch.save(self.model, os.path.join(self.best_model_save_path, "best_model.zip"))
                self.best_mean_reward = mean_reward

        return mean_reward, std_reward, sa_counts

    def _evaluate(self, noise, pi=None):
        eval_returns = []
        eval_successes = []
        sa_counts = np.zeros(shape=(self.eval_env.observation_space.shape[-1], self.eval_env.single_action_space.n))

        for episode_i in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            ep_returns = []
            ep_successes = []
            done = False
            step = 0
            while not done:
                step += 1
                # ALGO LOGIC: put action logic here
                if pi is not None:
                    s_idx = np.argmax(obs)
                    actions = [np.random.choice(np.arange(self.eval_env.single_action_space.n), p=pi[s_idx])]
                else:
                    with torch.no_grad():
                        actions = self.model.get_action(torch.Tensor(obs).to(self.device), noise=noise)
                        # actions = self.model(torch.Tensor(obs).to(self.device))
                        actions = actions.cpu().numpy()

                # s_idx = np.where(obs == 1)[1]
                # sa_counts[s_idx, actions] += 1

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, rewards, terminateds, truncateds, infos = self.eval_env.step(actions)
                done = terminateds[0] or truncateds[0]

                # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
                obs = next_obs

                ep_returns.append(rewards[0])
                ep_successes.append(terminateds[0])

            eval_returns.append(np.sum(ep_returns))
            eval_successes.append(np.sum(ep_successes) * 100)

        return eval_returns, eval_successes, sa_counts

    def simulate(self, train_env):
        self.eval_env = copy.deepcopy(train_env)

        eval_returns = []
        eval_successes = []
        sa_counts = np.zeros(shape=(self.eval_env.observation_space.shape[-1], self.eval_env.single_action_space.n))
        eval_obs = []
        eval_actions = []
        eval_rewards = []

        for episode_i in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            ep_returns = []
            ep_successes = []
            done = False
            step = 0
            # discounted_return = 0

            while not done:
                step += 1
                # ALGO LOGIC: put action logic here
                with torch.no_grad():
                    actions = self.model.get_action(torch.Tensor(obs).to(self.device), noise=False)
                    # actions = self.model(torch.Tensor(obs).to(self.device))
                    actions = actions.cpu().numpy()

                s_idx = np.where(obs == 1)[1]
                sa_counts[s_idx, actions] += 1

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, rewards, terminateds, truncateds, infos = self.eval_env.step(actions)
                done = terminateds[0] or truncateds[0]

                eval_obs.append(obs)
                eval_actions.append(actions)
                eval_rewards.append(rewards)
                # discounted_return += rewards

                # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
                obs = next_obs

                ep_returns.append(rewards[0])
                ep_successes.append(terminateds[0])

            # eval_returns.append(discounted_return)
            eval_returns.append(np.sum(ep_returns))
            eval_successes.append(np.sum(ep_successes) * 100)

        return np.array(eval_returns), np.array(eval_obs), np.array(eval_actions), np.array(eval_rewards), sa_counts