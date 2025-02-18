# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import copy
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass

import gymnasium as gym
import custom_envs
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro as tyro
import yaml

# import custom_envs
from stable_baselines3.common.utils import get_latest_run_id
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

# from PROPS.gridworld_clean.compute_true_gradient import compute_gradient
from collections import deque

from utils import simulate

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
    total_timesteps: int = 500000

    # Evaluation
    num_evals: int = 40
    eval_freq: int = 10
    eval_episodes: int = 0
    compute_sampling_error: bool = False

    # Architecture arguments
    linear: int = 0

    # Learning algorithm
    algo: str = 'ppo'

    # Sampling algorithm
    # sampling_algo: str = 'props'
    sampling_algo: str = 'on_policy'

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    learning_rate: float = 0
    num_envs: int = 1
    num_steps: int = 1024
    anneal_lr: bool = False
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 16
    update_epochs: int = 16
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
    props_num_minibatches: int = 4
    props_clip_coef: float = 0.3
    props_target_kl: float = 0.1
    props_lambda: float = 0.0
    props_freeze_features: bool = False

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

    def get_action_and_info(self, x, action=None):
        logits = self.actor(x)
        # logits = torch.clamp(logits, min=-3, max=1)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()

    def get_action(self, x):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action

    def get_logprob(self, x, action):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        return probs.log_prob(action)

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

            # kl_regularizer_loss = (torch.exp(b_logprobs[mb_inds])*(newlogprob - b_logprobs[mb_inds])).mean()


            # Policy loss
            pg_loss1 = ratio
            pg_loss2 = torch.clamp(ratio, 1 - args.props_clip_coef, 1 + args.props_clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            loss = pg_loss #- args.props_lambda * kl_regularizer_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
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

def compute_se(agent, b_obs, b_actions, envs):
    # COMPUTE SAMPLING ERROR

    # Initialize empirical policy equal to the current PPO policy.
    agent_mle = copy.deepcopy(agent)
    # agent_mle = Agent(envs, linear=False)
    # params = [p for p in agent.actor.parameters()]
    # print(params)

    # Freeze the feature layers of the empirical policy (as done in the Robust On-policy Sampling (ROS) paper)
    # params = [p for p in agent_mle.actor.parameters()]
    # params[0].requires_grad = False
    # params[1].requires_grad = False
    # params[2].requires_grad = False
    # params[3].requires_grad = False

    num_epochs = 5000
    optimizer_mle = optim.Adam(agent_mle.parameters(), lr=1e-3)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_mle, T_max=num_epochs,)
    n = len(b_obs)
    b_inds = np.arange(n)

    mb_size = 256 * n//256
    epoch = 0
    while True:
        epoch += 1
        np.random.shuffle(b_inds)
        for start in range(0, n, mb_size):
            end = start + mb_size
            mb_inds = b_inds[start:end]

            _, logprobs_mle, entropy_mle = agent_mle.get_action_and_info(b_obs[mb_inds], b_actions[mb_inds])
            loss = -torch.mean(logprobs_mle)# - 1*torch.mean(entropy_mle)

            optimizer_mle.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(agent_mle.parameters(), 1, norm_type=2)
            optimizer_mle.step()
        lr_scheduler.step()
        # print(lr_scheduler.optimizer.param_groups[0]["lr"])
        # print(grad_norm)
        # if loss < 0.01: break
        # print()
        # if epoch % 1000 == 0:
        if epoch % num_epochs == 0:
            print(loss, grad_norm, entropy_mle.mean())
            break


    with torch.no_grad():
        _, logprobs_mle, _ = agent_mle.get_action_and_info(b_obs, b_actions)
        _, logprobs_target, ent_target = agent.get_action_and_info(b_obs, b_actions)

        # Compute sampling error
        approx_kl_mle_target = (logprobs_mle - logprobs_target).mean()

        # logs = {}
        # logs[f'kl_mle_target'].append(approx_kl_mle_target.item())
        # logs[f'kl_props_target'].append(approx_kl_props_target.item())
        # logs[f'ent_target'].append(ent_target.mean().item())
        # logs[f'ent_props'].append(ent_props.mean().item())
    return approx_kl_mle_target.item()

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

    ### Seeding
    if args.run_id:
        args.seed = args.run_id
    elif args.seed is None:
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
    args.output_dir = f"{args.output_rootdir}/{args.env_id}/{args.algo}/{args.sampling_algo}/{args.output_subdir}"
    if args.run_id is not None:
        args.output_dir += f"/run_{args.run_id}"
    else:
        run_id = get_latest_run_id(log_path=args.output_dir, log_name='run_') + 1
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
    if args.props_freeze_features:
        params = [p for p in agent_props.actor.parameters()]
        for p in params[:4]:
            p.requires_grad = False

    optimizer_props = optim.Adam(agent_props.parameters(), lr=args.props_learning_rate, eps=1e-5)

    ### Logging
    logs = defaultdict(list)
    logs_sampling_error = defaultdict(list)

    target_update_count = 0

    # ALGO Logic: Storage setup
    obs_buffer = torch.zeros((args.buffer_size, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions_buffer = torch.zeros((args.buffer_size, args.num_envs) + envs.single_action_space.shape).to(device)
    rewards_buffer = torch.zeros((args.buffer_size, args.num_envs)).to(device)
    dones_buffer = torch.zeros((args.buffer_size, args.num_envs)).to(device)
    agent_buffer = deque(maxlen=args.buffer_size)
    envs_buffer = deque(maxlen=args.buffer_size)

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

        # agent_history.append(copy.deepcopy(agent))
        # envs_history.append(copy.deepcopy(envs))

        # shift buffers left by one batch. We will place the next batch we collect at the end of the buffer.
        # obs_buffer = torch.roll(obs_buffer, shifts=-args.num_steps, dims=0)
        # actions_buffer = torch.roll(actions_buffer, shifts=-args.num_steps, dims=0)
        # rewards_buffer = torch.roll(rewards_buffer, shifts=-args.num_steps, dims=0)
        # dones_buffer = torch.roll(dones_buffer, shifts=-args.num_steps, dims=0)

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

            # Once the buffer is full, we only write to last batch in the buffer for all subsequent collection phases.
            buffer_pos += 1
            if buffer_pos == args.buffer_size:
                buffer_pos = args.buffer_size - args.num_steps

            ################################## START BEHAVIOR UPDATE ##################################
            if args.sampling_algo in ['ros', 'props'] and global_step % args.props_num_steps == 0:
                obs = obs_buffer[:global_step]
                actions = actions_buffer[:global_step]
                with torch.no_grad():
                    logprobs = agent.get_logprob(obs, actions)

                b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
                b_actions = actions.reshape((-1,) + envs.single_action_space.shape).long()
                b_logprobs = logprobs.reshape(-1)

                props_update(agent_props, optimizer_props, b_obs, b_actions, b_logprobs, args)
            ################################## END BEHAVIOR UPDATE ##################################

        obs = obs_buffer[:global_step]
        actions = actions_buffer[:global_step]
        rewards = rewards_buffer[:global_step]
        dones = dones_buffer[:global_step]

        with torch.no_grad():
            values = agent.get_value(obs).reshape(-1, args.num_envs)
            logprobs = agent.get_logprob(obs, actions).reshape(-1, args.num_envs)

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

        ### Target policy (and value network) update
        # flatten buffer data
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape).long()
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        if args.compute_sampling_error:
            # b_actions = b_actions.reshape(-1)
            kl_mle_target = compute_se(agent, b_obs, b_actions, envs)
            logs['sampling_error'].append(kl_mle_target)
            print(logs['sampling_error'])
            # print(agent.actor)

        target_update_count += 1
        ppo_update(agent, optimizer, b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_values, args)
        ################################## END TARGET UPDATE ##################################

        if iteration % args.eval_freq == 0:
            return_avg, return_std, success_avg, success_std = simulate(env=env_eval, actor=agent, eval_episodes=args.eval_episodes)
            print(
                f"Eval num_timesteps={global_step}, " f"episode_return={return_avg:.2f} +/- {return_std:.2f}\n"
                f"Eval num_timesteps={global_step}, " f"episode_success={success_avg:.2f} +/- {success_std:.2f}\n"
            )

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