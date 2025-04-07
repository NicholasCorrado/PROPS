from collections import defaultdict

import numpy as np
import torch

def simulate(env, actor, eval_episodes, eval_steps=np.inf):
    logs = defaultdict(list)
    step = 0
    for episode_i in range(eval_episodes):
        logs_episode = defaultdict(list)

        obs, _ = env.reset()
        done = False

        while not done:

            # ALGO LOGIC: put action logic here
            with torch.no_grad():
                actions = actor.get_action(torch.Tensor(obs).to('cpu'), sample=False)
                actions = actions.cpu().numpy()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminateds, truncateds, infos = env.step(actions)
            done = np.logical_or(terminateds, truncateds)

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
    return return_avg, return_std, success_avg, success_std
    # return np.array(eval_returns), np.array(eval_obs), np.array(eval_actions), np.array(eval_rewards)

# def simulate_vec():
#     obs, _ = envs.reset()
#     episodic_returns = []
#     while len(episodic_returns) < eval_episodes:
#         actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
#         next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
#         if "final_info" in infos:
#             for info in infos["final_info"]:
#                 if "episode" not in info:
#                     continue
#                 print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
#                 episodic_returns += [info["episode"]["r"]]
#         obs = next_obs

def simulate_np(env, actor, eval_episodes, eval_steps=np.inf):
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


