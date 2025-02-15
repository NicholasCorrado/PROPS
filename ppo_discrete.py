# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import copy
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro as tyro
import yaml

# import custom_envs
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

# from PROPS.gridworld_clean.compute_true_gradient import compute_gradient
from utils import get_latest_run_id
from collections import deque

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False

    # Logging
    output_rootdir: str = 'results'
    output_subdir: str = ''
    run_id: int = None
    seed: int = 0
    total_timesteps: int = 50000

    # Evaluation
    eval_freq: int = 10
    eval_episodes: int = 20
    se_freq: int = None

    # Architecture arguments
    linear: int = 0

    # Learning algorithm
    algo: str = 'ppo'

    # Sampling algorithm
    sampling_algo: str = 'props'
    # sampling_algo: str = 'on_policy'

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    learning_rate: float = 1e-3
    num_envs: int = 1
    num_steps: int = 128
    anneal_lr: bool = False
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    buffer_size: int = 1

    # Behavior
    props_num_steps: int = 16
    props_learning_rate: float = 1e-3
    props_update_epochs: int = 4
    props_num_minibatches: int = 4
    props_clip_coef: float = 0.3
    props_target_kl: float = 0.03
    props_lambda: float = 0.1

    # to be filled in runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, linear):
        super().__init__()
        self.obs_dim = envs.single_observation_space.shape[0]

        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

        if linear:
            self.critic = nn.Sequential(
                layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 1), std=0),
            )
            self.actor = nn.Sequential(
                layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), envs.single_action_space.n),
                           std=0),
            )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def get_action(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()

    def get_pi_at_s(self, x):
        with torch.no_grad():
            logits = self.actor(x)
            probs = Categorical(logits=logits).probs.detach().numpy()
        return probs

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


def ppo_update(
        agent,  # Policy network (actor-critic architecture)
        optimizer,  # Optimizer (typically Adam)
        b_obs,  # Batch of observations (states)
        b_actions,  # Batch of actions taken
        b_logprobs,  # Batch of log probabilities of taken actions
        b_advantages,  # Batch of advantage estimates
        b_returns,  # Batch of returns (discounted rewards)
        b_values,  # Batch of value estimates
        args,  # Arguments containing hyperparameters
):
    """
    Performs a PPO policy update step using minibatching and clipped objectives.

    Args:
        agent: The actor-critic policy network that is being updated
        optimizer: The optimizer (typically Adam) used for updating the policy
        b_obs (torch.Tensor): Batch of observations/states from the environment
        b_actions (torch.Tensor): Batch of actions taken in the environment
        b_logprobs (torch.Tensor): Log probabilities of the actions taken under the old policy
        b_advantages (torch.Tensor): Computed advantage estimates for each timestep
        b_returns (torch.Tensor): Computed returns (discounted sum of rewards)
        b_values (torch.Tensor): Value estimates from the old policy
        args: Object containing PPO hyperparameters including:
            - num_minibatches (int): Number of minibatches to split the data into
            - update_epochs (int): Number of epochs to update on the same batch of data
            - clip_coef (float): PPO clipping coefficient (epsilon in the paper)
            - norm_adv (bool): Whether to normalize advantages
            - clip_vloss (bool): Whether to use clipped value loss
            - ent_coef (float): Entropy bonus coefficient
            - vf_coef (float): Value function loss coefficient
            - max_grad_norm (float): Maximum gradient norm for clipping
            - target_kl (float, optional): Target KL divergence threshold for early stopping
        target_update_count (int, optional): Counter for tracking target network updates
    """
    ### Target policy (and value network) update
    batch_size = len(b_obs)
    minibatch_size = max(batch_size // args.num_minibatches, batch_size)
    b_inds = np.arange(batch_size)
    clipfracs = []
    for epoch in range(args.update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_inds = b_inds[start:end]

            _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                # old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

            mb_advantages = b_advantages[mb_inds]
            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1)
            if args.clip_vloss:
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

        if args.target_kl is not None and approx_kl > args.target_kl:
            break

    # y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    # var_y = np.var(y_true)
    # explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    # writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
    # writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    # writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    # writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
    # writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
    # writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    # writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
    # writer.add_scalar("losses/explained_variance", explained_var, global_step)
    # writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    ################################## END TARGET UPDATE ##################################

def props_update(
        agent,  # Policy network (actor-critic architecture)
        optimizer,  # Optimizer (typically Adam)
        b_obs,  # Batch of observations (states)
        b_actions,  # Batch of actions taken
        b_logprobs,  # Batch of log probabilities of taken actions
        args,  # Arguments containing hyperparameters
):
    """
    Performs a PROPS policy update step using minibatching and clipped objectives.

    Args:
        agent: The actor-critic policy network that is being updated
        optimizer: The optimizer (typically Adam) used for updating the policy
        b_obs (torch.Tensor): Batch of observations/states from the environment
        b_actions (torch.Tensor): Batch of actions taken in the environment
        b_logprobs (torch.Tensor): Log probabilities of the actions taken under the old policy
        b_advantages (torch.Tensor): Computed advantage estimates for each timestep
        b_returns (torch.Tensor): Computed returns (discounted sum of rewards)
        b_values (torch.Tensor): Value estimates from the old policy
        args: Object containing PROPS hyperparameters including:
            - num_minibatches (int): Number of minibatches to split the data into
            - update_epochs (int): Number of epochs to update on the same batch of data
            - clip_coef (float): PROPS clipping coefficient (epsilon in the paper)
            - norm_adv (bool): Whether to normalize advantages
            - clip_vloss (bool): Whether to use clipped value loss
            - ent_coef (float): Entropy bonus coefficient
            - vf_coef (float): Value function loss coefficient
            - max_grad_norm (float): Maximum gradient norm for clipping
            - target_kl (float, optional): Target KL divergence threshold for early stopping
    """
    ### Target policy (and value network) update
    batch_size = len(b_obs)
    minibatch_size = max(batch_size // args.props_num_minibatches, batch_size)
    b_inds = np.arange(batch_size)
    clipfracs = []
    for epoch in range(args.props_update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_inds = b_inds[start:end]

            _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                # old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > args.props_clip_coef).float().mean().item()]

            # Policy loss
            pg_loss1 = ratio
            pg_loss2 = torch.clamp(ratio, 1 - args.props_clip_coef, 1 + args.props_clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            loss = pg_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.props_max_grad_norm)
            optimizer.step()

        if args.props_target_kl is not None and approx_kl > args.props_target_kl:
            break

    # y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    # var_y = np.var(y_true)
    # explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    # writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
    # writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    # writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    # writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
    # writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
    # writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    # writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
    # writer.add_scalar("losses/explained_variance", explained_var, global_step)
    # writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    ################################## END TARGET UPDATE ##################################


def simulate(env, actor, eval_episodes, eval_steps=np.inf):
    logs = defaultdict(list)
    sa_count = np.zeros(shape=(env.observation_space.shape[0], env.action_space.n))
    step = 0
    for episode_i in range(eval_episodes):
        logs_episode = defaultdict(list)

        obs, _ = env.reset()
        done = False

        while not done:

            # ALGO LOGIC: put action logic here
            with torch.no_grad():
                actions, _, _ = actor.get_action(torch.Tensor(obs).to('cpu'))
                actions = actions.cpu().numpy()

            s_idx = np.argmax(obs)
            a_idx = actions
            sa_count[s_idx, a_idx] += 1

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminateds, truncateds, infos = env.step(actions)
            done = terminateds or truncateds

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            logs_episode['rewards'].append(rewards)

            step += 1

            if step >= eval_steps:
                break
        if step >= eval_steps:
            break


        logs['returns'].append(np.sum(logs_episode['rewards']))
        try:
            logs['successes'].append(infos['is_success'])
        except:
            logs['successes'].append(False)

    return_avg = np.mean(logs['returns'])
    return_std = np.std(logs['returns'])
    success_avg = np.mean(logs['successes'])
    success_std = np.std(logs['successes'])
    return return_avg, return_std, success_avg, success_std, sa_count
    # return np.array(eval_returns), np.array(eval_obs), np.array(eval_actions), np.array(eval_rewards), sa_counts



def simulate_fast(env, actor, eval_episodes, eval_steps=np.inf):
    logs = defaultdict(list)
    sa_count = np.zeros(shape=(env.observation_space.shape[0], env.action_space.n))
    step = 0

    pi = actor.get_pi()
    for episode_i in range(eval_episodes):
        logs_episode = defaultdict(list)

        obs, _ = env.reset()
        done = False

        while not done:

            # ALGO LOGIC: put action logic here
            with torch.no_grad():
                s_idx = np.argmax(obs)
                pi_at_s = pi[s_idx]
                actions = np.random.choice(np.arange(env.action_space.n), p=pi_at_s)
                # print(pi_at_s, actions)

            a_idx = actions
            sa_count[s_idx, a_idx] += 1

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminateds, truncateds, infos = env.step(actions)
            done = terminateds or truncateds

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            logs_episode['rewards'].append(rewards)

            step += 1

            if step >= eval_steps:
                break
        if step >= eval_steps:
            break


        logs['returns'].append(np.sum(logs_episode['rewards']))
        try:
            logs['successes'].append(infos['is_success'])
        except:
            logs['successes'].append(False)

    return_avg = np.mean(logs['returns'])
    return_std = np.std(logs['returns'])
    success_avg = np.mean(logs['successes'])
    success_std = np.std(logs['successes'])
    return return_avg, return_std, success_avg, success_std, sa_count
    # return np.array(eval_returns), np.array(eval_obs), np.array(eval_actions), np.array(eval_rewards), sa_counts



def compute_se(args, agent, agent_props, obs, actions, global_step, envs):
    # COMPUTE SAMPLING ERROR

    # Initialize empirical policy equal to the current PPO policy.
    agent_mle = copy.deepcopy(agent)
    # agent_mle = Agent(envs, linear=False)

    # Freeze the feature layers of the empirical policy (as done in the Robust On-policy Sampling (ROS) paper)
    # params = [p for p in agent_mle.actor_mean.parameters()]
    # params[0].requires_grad = False
    # params[2].requires_grad = False

    optimizer_mle = optim.Adam(agent_mle.parameters(), lr=1e-3)
    obs_dim = obs.shape[-1]
    action_dim = actions.shape[-1]
    b_obs = obs.reshape(-1, obs_dim).to('cpu')
    b_actions = actions.reshape(-1).to('cpu')

    if global_step < args.buffer_size * args.num_steps:
        b_obs = b_obs[:global_step]
        b_actions = b_actions[:global_step]

    n = len(b_obs)
    b_inds = np.arange(n)

    mb_size = n
    for epoch in range(300):

        np.random.shuffle(b_inds)
        for start in range(0, n, mb_size):
            end = start + mb_size
            mb_inds = b_inds[start:end]

            _, logprobs_mle, _ = agent_mle.get_action(b_obs[mb_inds], b_actions[mb_inds])
            loss = -torch.mean(logprobs_mle)

            optimizer_mle.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(agent_mle.parameters(), 0.5, norm_type=2)
            optimizer_mle.step()
            # print(grad_norm)
            # if grad_norm < 1e-5:
            #     print('break')
            #     break

            # print((logprobs_mle - logprobs_target).mean())
        # if (epoch+1) % 100 == 0:
        #     _, logprobs_mle, _ = agent_mle.get_action(b_obs, b_actions)
        #     _, logprobs_target, ent_target = agent.get_action(b_obs, b_actions)
        #     _, logprobs_props, ent_props = agent_props.get_action(b_obs, b_actions)
        #     approx_kl_mle_target = (logprobs_mle - logprobs_target).mean()
        #     print(epoch, approx_kl_mle_target)

    with torch.no_grad():
        _, logprobs_mle, _ = agent_mle.get_action(b_obs, b_actions)
        _, logprobs_target, ent_target = agent.get_action(b_obs, b_actions)
        # _, logprobs_props, ent_props = agent_props.get_action(b_obs, b_actions)

        # Compute sampling error
        approx_kl_mle_target = (logprobs_mle - logprobs_target).mean()

        # logs = {}
        # logs[f'kl_mle_target'].append(approx_kl_mle_target.item())
        # logs[f'kl_props_target'].append(approx_kl_props_target.item())
        # logs[f'ent_target'].append(ent_target.mean().item())
        # logs[f'ent_props'].append(ent_props.mean().item())
    return approx_kl_mle_target.item()


def update_behavior_policy(args, global_step, envs, obs, logprobs, actions, agent_props, agent, optimizer_props):
    # behavior_update += 1

    ### Flatten the batch
    b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + envs.single_action_space.shape)

    # @TODO: fix when you add historic data
    if global_step < args.buffer_size * args.num_steps:
        b_obs = b_obs[:global_step]
        b_logprobs = b_logprobs[:global_step]
        b_actions = b_actions[:global_step]

    ### Set props policy equal to current target policy
    for source_param, dump_param in zip(agent_props.parameters(), agent.parameters()):
        source_param.data.copy_(dump_param.data)
    # Freeze the feature layers of the empirical policy (as done in the Robust On-policy Sampling (ROS) paper)
    # params = [p for p in agent_props.actor.parameters()]
    # for p in params[:4]:
    #     p.requires_grad = False

    props_batch_size = len(b_obs)
    props_minibatch_size = max(int(props_batch_size // args.props_num_minibatches), props_batch_size)
    b_inds = np.arange(props_batch_size)
    clipfracs = []
    for epoch in range(args.props_update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, props_batch_size, props_minibatch_size):
            end = start + props_minibatch_size
            mb_inds = b_inds[start:end]

            _, newlogprob, entropy, newvalue = agent_props.get_action_and_value(b_obs[mb_inds],
                                                                                b_actions.long()[mb_inds])
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > args.props_clip_coef).float().mean().item()]

            # @TODO: add KL regularization
            mb_advantages = -1

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            kl_loss = ((ratio - 1) - logratio).mean()

            loss = pg_loss  # + args.props_lambda * kl_loss

            optimizer_props.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(agent_props.parameters(), args.max_grad_norm)
            optimizer_props.step()

        if args.props_target_kl is not None and approx_kl > args.props_target_kl:
            break


def update_behavior_policy2(agent_props, envs, props_optimizer, obs, actions, advantages, global_step, args):
    # PROPS UPDATE


    if global_step <= args.buffer_size - args.props_num_steps:
        # If the replay buffer is not full, use all data in replay buffer for this update.
        start = 0
        end = global_step
    else:
        # If the replay buffer is full, exclude the oldest behavior batch from this update; that batch will be evicted
        # before the next update and thus does not contribute to sampling error.
        start = args.props_num_steps
        end = args.buffer_size
    # flatten the replay buffer data
    b_obs = obs[start:end].reshape((-1,) + envs.single_observation_space.shape).to(args.device)
    b_actions = actions[start:end].reshape((-1,) + envs.single_action_space.shape).to(args.device)
    # b_logits = logits[start:end].reshape(-1)  # action logits for PPO policy
    with torch.no_grad():
        _, _, logprobs, _, _ = agent_props.get_action_and_value(b_obs, b_actions)
    b_logprobs = logprobs.reshape(-1).to(args.device)

    b_probs = torch.exp(logprobs).to(args.device)

    if args.props_adv:
        b_advantages = advantages[start:end].reshape(-1)

    batch_size = b_obs.shape[0]
    minibatch_size = min(args.props_minibatch_size, batch_size)
    b_inds = np.arange(batch_size)
    clipfracs = []

    done_updating = False
    num_update_minibatches = 0
    pg_loss = None
    kl_regularizer_loss = None
    approx_kl_to_log = None
    grad_norms = []

    for epoch in range(args.props_update_epochs):
        np.random.shuffle(b_inds)

        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_inds = b_inds[start:end]
            mb_obs = b_obs[mb_inds]
            mb_actions = b_actions[mb_inds]
            mb_probs = b_probs[mb_inds]
            mb_logprobs = b_logprobs[mb_inds]

            if args.props_adv:
                # Do not zero-center advantages; we need to preserve A(s,a) = 0 for AW-PROPS
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - 0) / (mb_advantages.std() + 1e-8)
                mb_abs_advantages = torch.abs(mb_advantages)
                # print(torch.mean(mb_abs_advantages), torch.std(mb_abs_advantages))

            _, _, props_logprobs, entropy = agent_props.get_action_and_info(mb_obs, mb_actions)
            props_logratio = props_logprobs - b_logprobs[mb_inds]
            props_ratio = props_logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-props_logratio).mean()
                approx_kl = ((props_ratio - 1) - props_logratio).mean()
                clipfracs += [((props_ratio - 1.0).abs() > args.props_clip_coef).float().mean().item()]


                approx_kl_to_log = approx_kl

            kl_regularizer_loss = (mb_probs*(mb_logprobs - props_logprobs)).mean()

            pg_loss1 = props_ratio
            pg_loss2 = torch.clamp(props_ratio, 1 - args.props_clip_coef, 1 + args.props_clip_coef)
            if args.props_adv:
                pg_loss = (torch.max(pg_loss1, pg_loss2) * mb_abs_advantages).mean()
            else:
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            if args.ros:
                pg_loss = props_logratio.mean()

            entropy_loss = entropy.mean()
            loss = pg_loss + args.props_lambda * kl_regularizer_loss

            props_optimizer.zero_grad()
            loss.backward()

            grad_norm = nn.utils.clip_grad_norm_(agent_props.parameters(), args.props_max_grad_norm)
            grad_norms.append(grad_norm.detach().cpu().numpy())

            props_optimizer.step()
            num_update_minibatches += 1

        if args.props_target_kl:
            # print(approx_kl)
            if approx_kl > args.props_target_kl:
                done_updating = True
                break

        if done_updating:
            break
def run():
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    args.props_batch_size = int(args.num_envs * args.props_num_steps)
    args.props_minibatch_size = int(args.props_batch_size // args.props_num_minibatches)
    props_iterations_per_target_update = args.num_steps // args.props_num_steps

    if args.sampling_algo in ['props', 'ros']:
        assert args.num_steps % args.props_num_steps == 0

    ### Seeding
    if args.seed is None:
        if args.run_id:
            args.seed = np.random.randint(args.run_id)
        else:
            args.seed = np.random.randint(2 ** 32 - 1)

    ### Override hyperparameters based on sampling method
    assert args.sampling_algo in ['on_policy', 'ros', 'props', 'greedy_adaptive', 'oracle_adaptive']
    if args.algo == 'ros':
        args.props_num_steps = 1
        args.props_update_epochs = 1
        args.props_clip_coef = 9999999
        args.props_target_kl = 9999999
        args.props_lambda = 0

    ### Output path
    args.output_dir = f"{args.output_rootdir}/{args.env_id}/{args.algo}_{args.sampling_algo}/{args.output_subdir}"
    if args.run_id is not None:
        args.output_dir += f"/run_{args.run_id}"
    else:
        run_id = get_latest_run_id(save_dir=args.output_dir) + 1
        args.output_dir += f"/run_{run_id}"

    ### Dump training config to save dir
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.yml"), "w") as f:
        yaml.dump(args, f, sort_keys=True)

    ### wandb
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    # writer = SummaryWriter(f"runs/{run_name}")
    # writer.add_text(
    #     "hyperparameters",
    #     "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    # )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    env_eval = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    ).envs[0]

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs, args.linear).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    agent_props = copy.deepcopy(agent)
    # Freeze the feature layers of the empirical policy (as done in the Robust On-policy Sampling (ROS) paper)
    # params = [p for p in agent_props.actor.parameters()]
    # for p in params[:4]:
    #     p.requires_grad = False

    optimizer_props = optim.Adam(agent_props.parameters(), lr=args.props_learning_rate, eps=1e-5)

    ### Logging
    logs = defaultdict(list)
    logs_sampling_error = defaultdict(list)

    target_update_count = 0
    behavior_update_count = 0

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.buffer_size * args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.buffer_size * args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.buffer_size * args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.buffer_size * args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.buffer_size * args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.buffer_size * args.num_steps, args.num_envs)).to(device)
    agent_history = deque(maxlen=args.buffer_size)
    envs_history = deque(maxlen=args.buffer_size)

    ### Oracle adaptive sampling setup
    sa_counts = np.zeros(shape=(envs.single_observation_space.shape[-1], envs.single_action_space.n))
    possible_actions = np.arange(envs.single_action_space.n)

    ### Load exact gradient
    if 'Chain' in args.env_id or 'GridWorld' in args.env_id:
        # sa_occupancy_true = np.load(f'gridworld_clean/data/{args.env_id}/non_uniform/sa_occupancy_true.npy')
        # grad_true = np.load(f'gridworld_clean/data/{args.env_id}/non_uniform/grad_true.npy')
        # adv_true = np.load(f'gridworld_clean/data/{args.env_id}/non_uniform/adv_true.npy')
        sa_occupancy_true = np.load(f'gridworld_clean/data/{args.env_id}/sa_occupancy_true.npy')
        grad_true = np.load(f'gridworld_clean/data/{args.env_id}/grad_true.npy')
        adv_true = np.load(f'gridworld_clean/data/{args.env_id}/adv_true.npy')
        grad_true_norm = np.linalg.norm(grad_true)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    buffer_pos = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        agent_history.append(copy.deepcopy(agent))
        envs_history.append(copy.deepcopy(envs))
        # sa_counts[:] = 0

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[buffer_pos] = next_obs
            dones[buffer_pos] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                if args.sampling_algo in ['props', 'ros']:
                    action, _, _, _ = agent_props.get_action_and_value(next_obs)
                    _, logprob, _, value = agent.get_action_and_value(next_obs, action=action)
                    a_idx = action[0].item()
                elif args.sampling_algo == 'greedy_adaptive':
                    s_idx = np.argmax(next_obs)
                    # print(s_idx)
                    sa = sa_counts[s_idx]
                    pi = agent.get_pi_at_s(next_obs)[0]
                    # pi = np.array([0.5, 0.5])
                    if np.sum(sa) == 0:
                        a_idx = np.random.choice(possible_actions, p=pi)
                    else:
                        pi_empirical = sa / np.sum(sa)
                        a_idx = np.argmin(pi_empirical - agent.get_pi_at_s(next_obs))

                    action = torch.Tensor([a_idx])
                    _, logprob, _, value = agent.get_action_and_value(next_obs, action)
                elif args.sampling_algo == 'oracle_adaptive':
                    raise NotImplementedError()
                else:
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    a_idx = action[0].item()
                    # a_idx = np.random.choice(possible_actions, p=[0.1, 0.1, 0.1, 0.7])

                values[buffer_pos] = value.flatten()
            actions[buffer_pos] = action
            logprobs[buffer_pos] = logprob

            # s_idx = np.where(next_obs[0] == 1)[0][0]
            # sa_counts[s_idx, a_idx] += 1

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[buffer_pos] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            buffer_pos += 1
            buffer_pos %= args.buffer_size * args.num_steps
            # buffer_pos = np.clip(buffer_pos, a_min=0, a_max=args.num_steps-1)
            # if "final_info" in infos:
            #     for info in infos["final_info"]:
            #         if info and "episode" in info:
                        # print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        # writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        # writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

            if args.se_freq and global_step % args.se_freq == 0:
                if 'Chain' in args.env_id or 'GridWorld' in args.env_id:

                    b_obs = obs[:buffer_pos].reshape((-1,) + envs.single_observation_space.shape)
                    b_actions = actions[:buffer_pos].reshape((-1,) + envs.single_action_space.shape)
                    b_obs = b_obs.detach().numpy()
                    b_actions = b_actions.detach().numpy()

                    # print(global_step, buffer_pos)
                    # sa_counts[:] = 0
                    for o, a in zip(b_obs, b_actions):
                        s_idx = np.where(o == 1)[0][0]
                        # a_idx = a[0]
                        a_idx = int(a)
                        sa_counts[s_idx, a_idx] += 1

                    grad_empirical = compute_gradient(envs.envs[0].unwrapped, agent.get_pi(), b_obs, b_actions.astype(int), adv_true)
                    # grad_empirical_norm = np.linalg.norm(grad_empirical)
                    grad_accuracy = (grad_empirical @ grad_true) / np.linalg.norm(grad_empirical) / grad_true_norm

                    logs_sampling_error['grad'].append(grad_empirical)
                    logs_sampling_error['grad_true'].append(grad_true)
                    logs_sampling_error['grad_accuracy'].append(grad_accuracy)

                    sa_occupancy = sa_counts / sa_counts.sum()
                    se = np.abs(sa_occupancy - sa_occupancy_true).sum()
                    # print(sa_occupancy)

                    logs_sampling_error['sampling_error'].append(se)
                    logs_sampling_error['sa_occupancy'].append(sa_occupancy)
                    logs_sampling_error['sa_occupancy_true'].append(sa_occupancy_true)

                    # print(se)
                else:
                    kl_mle_target = compute_se(args, agent, agent_props, obs, actions, global_step, envs)
                    logs_sampling_error['sampling_error'].append(kl_mle_target)

                logs_sampling_error['timestep'].append(global_step)
                np.savez(
                    f'{args.output_dir}/sampling_error.npz',
                    **logs_sampling_error,
                )

            ################################## START BEHAVIOR UPDATE ##################################
            if args.sampling_algo in ['ros', 'props'] and global_step % args.props_num_steps == 0:
                props_update(agent_props, optimizer_props, b_obs, b_actions, b_logprobs, args)
            ################################## END BEHAVIOR UPDATE ##################################

        # bootstrap value if not done
        # @TODO: make sure this runs over the most recent batch
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        ### Flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        if global_step < args.buffer_size * args.num_steps:
            b_obs = b_obs[:global_step]
            b_logprobs = b_logprobs[:global_step]
            b_actions = b_actions[:global_step]
            b_advantages = b_advantages[:global_step]
            b_returns = b_returns[:global_step]
            b_values = b_values[:global_step]
        elif args.buffer_size > 1:
            b_obs = torch.roll(b_obs, buffer_pos)
            b_logprobs = torch.roll(b_logprobs, buffer_pos)
            b_actions = torch.roll(b_actions, buffer_pos)
            b_advantages = torch.roll(b_advantages, buffer_pos)
            b_returns = torch.roll(b_returns, buffer_pos)
            b_values = torch.roll(b_values, buffer_pos)

        ### Target policy (and value network) update
        target_update_count += 1
        ppo_update(agent, optimizer, b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_values, args)
        ################################## END TARGET UPDATE ##################################


        if iteration % args.eval_freq == 0:
            return_avg, return_std, success_avg, success_std, _ = \
                simulate(env=env_eval, actor=agent, eval_episodes=args.eval_episodes)
            #
            # # collect on-policy data
            # _, _, _, _, sa_counts_on_policy = \
            #     simulate(env=copy.deepcopy(envs_history[-1].envs[0]), actor=agent_history[-1], eval_episodes=1000, eval_steps=args.num_steps)
            #
            # # collect a lot of on-policy data to approximate the true distribution
            # _, _, _, _, sa_counts_true = \
            #     simulate_fast(env=copy.deepcopy(envs_history[-1].envs[0]), actor=agent_history[-1], eval_episodes=10000)
            #
            #

            # sa_occupancy_true = sa_counts_true / sa_counts_true.sum()
            # sa_occupancy_on_policy = sa_counts_on_policy / sa_counts_on_policy.sum()
            # sa_occupancy = sa_counts / sa_counts.sum()
            #
            # se = np.abs(sa_occupancy - sa_occupancy_true).sum()
            # se_on_policy = np.abs(sa_occupancy_on_policy - sa_occupancy_true).sum()
            #
            # # print(sa_occupancy)
            #
            # print(se)
            # print(se_on_policy)
            # logs_sampling_error['sampling_error'].append(se)
            # logs_sampling_error['sampling_error_on_policy'].append(se_on_policy)
            # # logs['sa_occupancy_eval'].append(sa_counts_eval)
            # # logs_sampling_error['sa_occupancy'].append(sa_occupancy)
            # # logs_sampling_error['sa_occupancy_true'].append(sa_occupancy_true)

            print(f"Eval num_timesteps={global_step}, " f"episode_return={return_avg:.2f} +/- {return_std:.2f}")
            print(f"Eval num_timesteps={global_step}, " f"episode_success={success_avg:.2f} +/- {success_std:.2f}")
            print()

            logs['timestep'].append(global_step)
            logs['return'].append(return_avg)
            logs['success_rate'].append(success_avg)
            logs['target_update'].append(target_update_count)

            np.savez(
                f'{args.output_dir}/evaluations.npz',
                **logs,
                **logs_sampling_error
            )

    envs.close()
    # writer.close()




if __name__ == "__main__":
    run()