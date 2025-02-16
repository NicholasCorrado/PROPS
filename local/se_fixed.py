import os

buffer_batches = 8
num_steps = 256
total_timesteps = buffer_batches*num_steps
for env_id in [
    'CartPole-v1',
    # 'LunarLander-v3',
    # 'Discrete2D100-v0'
]:
    # for i in range(5):
    #     os.system(
    #         f"python ../ppo_discrete.py --run_id {i} --seed {i}"
    #         f" --env_id {env_id}"
    #         f" --buffer_batches {buffer_batches}"
    #         f" --num_steps {num_steps}"
    #         f" --learning_rate 0"
    #         f" --total_timesteps {total_timesteps}"
    #         f" --eval_freq 1"
    #         f" --eval_episodes 0"
    #         f" --compute_sampling_error"
    #         f" --sampling_algo on_policy"
    #     )

    for i in range(1):
        os.system(
            f"python ../ppo_discrete.py --run_id {i} --seed {i}"
            f" --env_id {env_id}"
            f" --buffer_batches {buffer_batches}"
            f" --num_steps {num_steps}"
            f" --learning_rate 0"
            f" --total_timesteps {total_timesteps}"
            f" --eval_freq 1"
            f" --eval_episodes 0"
            f" --compute_sampling_error"
            f" --sampling_algo props"
        )