import os
'''
Hyperparam sweep for AntMaze_{UMaze, Medium, Large}-v3, 5 seeds
'''

# python3 ddpg.py --env_id PandaSlide-v3 --daf RelabelGoal --save_subdir 2xTranslateGoal 

def gen_command_best(env_id, daf, timesteps, aug_r, subdir):
    python_command = f'python3 ddpg.py --env_id {env_id} ' \
                    f'--gamma 0.99 ' \
                    f'--exploration_noise 0.1 ' \
                    f'--learning_rate 1e-4 ' \
                    f'--batch_size 64 ' \
                    f'--net_arch 256 256 256 ' \
                    f'--aug_ratio {aug_r} ' \
                    f'--save_subdir {subdir} ' \
                    f'--total_timesteps {timesteps} ' \
                    f'--env_kwargs continuing_task False ' 
    if daf != None:
        python_command = python_command + f'--daf {daf} ' 
        python_command = python_command + f'--buffer_size {int(2e6)}' # x2 buffer
    else: # no DAF
        python_command = python_command + f'--buffer_size {int(1e6)}' # buffer=1mill

    mem = 7.5
    disk = 9.5
    command = f"{mem},{disk},{python_command}"
    return command

def gen_command_batch(env_id, daf, learning_rate, batch_size):
    python_command = f'python3 ddpg.py --env_id {env_id} ' \
                    f'--learning_rate {learning_rate} ' \
                    f'--batch_size {batch_size} ' \
                    f'--save_subdir lr_{learning_rate}/batch_{batch_size} ' \
                    f'--env_kwargs continuing_task False '
    if daf != None:
        python_command = python_command + f'--daf {daf} ' 
        python_command = python_command + f'--buffer_size {int(2e6)}' # x2 buffer
    else: # no DAF
        python_command = python_command + f'--buffer_size {int(1e6)}' # buffer=1M

    mem = 8
    disk = 10
    command = f"{mem},{disk},{python_command}"
    return command

def gen_command_net(env_id, learning_rate, batch_size, network, network_subdir):
    python_command = f'python3 ddpg.py --env_id {env_id} ' \
                    f'--learning_rate {learning_rate} ' \
                    f'--batch_size {batch_size} ' \
                    f'--net_arch {network} ' \
                    f'--save_subdir lr_{learning_rate}/tau_ ' \
                    f'--env_kwargs continuing_task False ' \
                    f'--total_timesteps 1000000 ' \
                    f'--buffer_size {int(1e6)}'

    mem = 10
    disk = 10
    command = f"{mem},{disk},{python_command}"
    return command

# 4/28
def gen_command_tau(env_id, learning_rate, tau):
    python_command = f'python3 ddpg.py --env_id {env_id} ' \
                    f'--random_action_prob 0 ' \
                    f'--learning_starts 10000 ' \
                    f'--learning_rate {learning_rate} ' \
                    f'--batch_size 64 ' \
                    f'--net_arch 256 256 256 ' \
                    f'--save_subdir lr_{learning_rate}/tau_{tau} ' \
                    f'--env_kwargs continuing_task False ' \
                    f'--total_timesteps 1000000 ' \
                    f'--buffer_size {int(1e6)}'

    mem = 10
    disk = 10
    command = f"{mem},{disk},{python_command}"
    return command

if __name__ == "__main__":

    os.makedirs('../commands', exist_ok=True)
    f = open(f"../commands/train_ant_sweep.txt", "w")
    
    # pointmaze_dafs = ['RelabelGoal', 'TranslateRotate', 'TranslateRotateRelabelGoal']
    batch_sizes = [64, 128, 256]

    # env_ids = ['AntMaze_UMaze-v4', 'AntMaze_Medium-v4', 'AntMaze_Large-v4']
    env_ids = ['AntMaze_UMaze-v4', 'AntMaze_UMazeDense-v4']
    # pointmaze_dafs = ['RelabelGoal']
    # buff = 2e6
    # aug_ratios = [1,2,4,8,16]
    # alpha = 0.5 # fixed alpha
    timesteps = 1000000
    net_archs = ["64 64", "256 256", "256 256 256"]
    net_arch_str = ["64,64", "256,256", "256,256,256"]

    # sweep over batch size with daf RelabelGoal
    # for env_id in env_ids: 
    #     for lr  in learning_rates:
    #         for batch in batch_sizes:
    #             command = gen_command_batch(env_id, daf=None, learning_rate=lr, batch_size=batch)
    #             print(command)
    #             f.write(command.replace(' ', '*') + "\n")

    #             for network, subdir in zip(net_archs,net_arch_str):
    #                 command = gen_command_net(env_id, lr, batch, network, subdir)
    #                 print(command)
    #                 f.write(command.replace(' ', '*') + "\n")

    # lr + tau sweep
    learning_rates = [1e-4, 1e-5]
    taus= [0.05, 0.005]
    env_ids = ['AntMaze_UMaze-v4', 'AntMaze_UMazeDense-v4']

    for env_id in env_ids: 
        for lr  in learning_rates:
            for tau in taus:
                command = gen_command_tau(env_id, lr, tau)
                print(command)
                f.write(command.replace(' ', '*') + "\n")

    