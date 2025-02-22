import copy
import os
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass

import gymnasium as gym
import custom_envs
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
import yaml
from gymnasium.wrappers.normalize import RunningMeanStd
from stable_baselines3.common.utils import get_latest_run_id
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from utils import simulate
from wrappers import NormalizeObservation, NormalizeReward


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
    seed: int = None
    total_timesteps: int = 1000000

    # Evaluation
    num_evals: int = 20
    eval_freq: int = 10
    eval_episodes: int = 20
    compute_sampling_error: bool = False

    # Architecture arguments
    linear: int = 0

    # Learning algorithm
    algo: str = 'ppo'

    # Sampling algorithm
    sampling_algo: str = 'on_policy'

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    learning_rate: float = 1e-3
    num_envs: int = 1
    num_steps: int = 2048
    anneal_lr: bool = False
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    buffer_batches: int = 1

    # Behavior
    props_num_steps: int = 8
    props_learning_rate: float = 1e-3
    props_update_epochs: int = 16
    props_num_minibatches: int = 16
    props_clip_coef: float = 0.3
    props_target_kl: float = 0.1
    props_lambda: float = 0.0
    props_freeze_features: bool = False

    # to be filled in runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        # env = gym.wrappers.NormalizeObservation(env)
        env = NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        # env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        return env

    return thunk

def make_eval_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, linear=False):
        super().__init__()
        if not linear:
            self.critic = nn.Sequential(
                layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 1), std=1.0),
            )
            self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
            )
        else:
            self.critic = nn.Sequential(
                layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 1), std=0),
            )
            self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(),
                                     np.prod(envs.single_action_space.shape)), std=0),
            )

        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))
        self.obs_dim = envs.single_observation_space.shape[0]

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

    def get_action_and_info(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1)

    def get_action(self, x, sample=True):
        action_mean = self.actor_mean(x)
        if sample:
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            probs = Normal(action_mean, action_std)
            action = probs.sample()
        else:
            action = action_mean
        return action

    def get_logprob(self, x, action):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.log_prob(action).sum(1)


def props_update(
        agent,
        optimizer,
        b_obs,
        b_actions,
        b_logprobs,
        args,
):
    batch_size = len(b_obs)
    minibatch_size = max(batch_size // args.props_num_minibatches, batch_size)
    b_inds = np.arange(batch_size)
    clipfracs = []

    for epoch in range(args.props_update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_inds = b_inds[start:end]

            _, newlogprob, entropy, _ = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > args.props_clip_coef).float().mean().item()]

            # Policy loss
            pg_loss1 = ratio
            pg_loss2 = torch.clamp(ratio, 1 - args.props_clip_coef, 1 + args.props_clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            loss = pg_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

        if args.props_target_kl is not None and approx_kl > args.props_target_kl:
            break

    logs = {
        'policy_loss': pg_loss.item(),
        'approx_kl': approx_kl.item(),
        'clipfrac': np.mean(clipfracs),
        'epoch': epoch
    }
    return logs


def run():
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    args.eval_freq = max(args.num_iterations // args.num_evals, 1)

    args.props_batch_size = int(args.num_envs * args.props_num_steps)
    args.props_minibatch_size = int(args.props_batch_size // args.props_num_minibatches)
    props_iterations_per_target_update = args.num_steps // args.props_num_steps

    args.buffer_size = args.buffer_batches * args.num_steps

    if args.sampling_algo in ['props', 'ros']:
        assert args.num_steps % args.props_num_steps == 0

    # Seeding
    if args.run_id:
        args.seed = args.run_id
    elif args.seed is None:
        args.seed = np.random.randint(2 ** 32 - 1)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Output path setup
    args.output_dir = f"{args.output_rootdir}/{args.env_id}/{args.algo}/{args.sampling_algo}/{args.output_subdir}"
    if args.run_id is not None:
        args.output_dir += f"/run_{args.run_id}"
    else:
        run_id = get_latest_run_id(log_path=args.output_dir, log_name='run_') + 1
        args.output_dir += f"/run_{run_id}"

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.yml"), "w") as f:
        yaml.dump(args, f, sort_keys=True)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    run_name = 'tmp'

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    envs_eval = gym.vector.SyncVectorEnv(
        [make_eval_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(1)]
    )
    normalize_obs = envs.envs[0].env.env.env
    normalize_reward = envs.envs[0].env

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs, args.linear).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    agent_props = copy.deepcopy(agent)

    if args.props_freeze_features:
        params = [p for p in agent_props.actor_mean.parameters()]
        for p in params[:4]:
            p.requires_grad = False

    optimizer_props = optim.Adam(agent_props.parameters(), lr=args.props_learning_rate, eps=1e-5)

    # Logging
    logs = defaultdict(list)

    # Storage setup
    obs_buffer = torch.zeros((args.buffer_size, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions_buffer = torch.zeros((args.buffer_size, args.num_envs) + envs.single_action_space.shape).to(device)
    rewards_buffer = torch.zeros((args.buffer_size, args.num_envs)).to(device)
    dones_buffer = torch.zeros((args.buffer_size, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    buffer_pos = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        if global_step >= args.buffer_size:
            obs_buffer = torch.roll(obs_buffer, shifts=-args.num_steps, dims=0)
            actions_buffer = torch.roll(actions_buffer, shifts=-args.num_steps, dims=0)
            rewards_buffer = torch.roll(rewards_buffer, shifts=-args.num_steps, dims=0)
            dones_buffer = torch.roll(dones_buffer, shifts=-args.num_steps, dims=0)

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs_buffer[buffer_pos] = next_obs
            dones_buffer[buffer_pos] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                if args.sampling_algo in ['props', 'ros']:
                    action, _, _, _ = agent_props.get_action_and_value(next_obs)
                    _, logprob, _, value = agent.get_action_and_value(next_obs, action=action)
                else:
                    action = agent.get_action(next_obs)

                actions_buffer[buffer_pos] = action

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                next_done = np.logical_or(terminations, truncations)
                rewards_buffer[buffer_pos] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")

            buffer_pos += 1
            if buffer_pos == args.buffer_size:
                buffer_pos = args.buffer_size - args.num_steps

            ################################## START BEHAVIOR UPDATE ##################################
            log_props = {}
            if args.sampling_algo in ['ros', 'props'] and global_step % args.props_num_steps == 0:
                obs = obs_buffer[:global_step]
                actions = actions_buffer[:global_step]
                with torch.no_grad():
                    logprobs = agent.get_logprob(obs, actions)

                b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
                b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
                b_logprobs = logprobs.reshape(-1)

                for source_param, dump_param in zip(agent_props.parameters(), agent.parameters()):
                    source_param.data.copy_(dump_param.data)

                log_props = props_update(agent_props, optimizer_props, b_obs, b_actions, b_logprobs, args)
            ################################## END BEHAVIOR UPDATE ##################################

        obs = obs_buffer[:global_step]
        actions = actions_buffer[:global_step]
        rewards = rewards_buffer[:global_step]
        dones = dones_buffer[:global_step]

        with torch.no_grad():
            values = agent.get_value(obs).reshape(-1, args.num_envs)
            logprobs = agent.get_logprob(obs, actions).reshape(-1, args.num_envs)

        # bootstrap value if not done
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

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
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
        #
        if iteration % args.eval_freq == 0:
            envs_eval = copy.deepcopy(envs)
            envs_eval.envs[0].set_eval()

            return_avg, return_std, success_avg, success_std = simulate(env=envs_eval, actor=agent, eval_episodes=args.eval_episodes)
            print(
                f"Eval num_timesteps={global_step}, "
                f"episode_return={return_avg:.2f} +/- {return_std:.2f}\n"
                f"episode_success={success_avg:.2f} +/- {success_std:.2f}\n"
            )

            logs['timestep'].append(global_step)
            logs['return'].append(return_avg)
            logs['success_rate'].append(success_avg)
            for key, value in log_props.items():
                logs[key].append(value)

            np.savez(
                f'{args.output_dir}/evaluations.npz',
                **logs,
            )

    envs.close()

if __name__ == "__main__":
    run()