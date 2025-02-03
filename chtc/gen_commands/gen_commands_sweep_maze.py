import os
'''
Hyperparam sweep for PointMaze_{UMaze, Medium, Large}-v3, 5 seeds 
'''

def gen_command_best(env_id, daf, timesteps, aug_r, subdir): # most recent change train_freq=1
    python_command = f'python3 ddpg.py --env_id {env_id} ' \
                    f'--train_freq 1 ' \
                    f'--gamma 0.99 ' \
                    f'--alpha 0.5 ' \
                    f'--exploration_noise 0.1 ' \
                    f'--learning_rate 0.0001 ' \
                    f'--batch_size 64 ' \
                    f'--net_arch 256 256 256 ' \
                    f'--save_subdir {subdir} ' \
                    f'--total_timesteps {timesteps} ' \
                    f'--env_kwargs continuing_task False ' 
    if aug_r != None:
        python_command = python_command + f'--aug_ratio {aug_r} ' 
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
                    f'--save_subdir lr_{learning_rate}_batch_{batch_size}/network_{network_subdir} ' \
                    f'--env_kwargs continuing_task False ' \
                    f'--total_timesteps 1000000 ' \
                    f'--buffer_size {int(1e6)}'

    mem = 10
    disk = 10
    command = f"{mem},{disk},{python_command}"
    return command

if __name__ == "__main__":

    os.makedirs('commands', exist_ok=True)
    f = open(f"commands/pointmaze_ar_sweep.txt", "w")
    
    env_ids = ['PointMaze_UMaze-v3', 'PointMaze_Medium-v3', 'PointMaze_Large-v3']
    pointmaze_dafs = ['RelabelGoal', 'TranslateRotate', 'TranslateRotateRelabelGoal']
    learning_rates = [1e-3, 1e-4]
    batch_sizes = [64, 128, 256]

    # env_ids = ['AntMaze_UMaze-v4', 'AntMaze_Medium-v4', 'AntMaze_Large-v4']
    # buff = 2e6
    aug_ratios = [1,4,8,16]
    # alpha = 0.5 # fixed alpha
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
                    # f.write(command.replace(' ', '*') + "\n")

    # sweep aug_r
    for daf in pointmaze_dafs:
        for aug_r in aug_ratios:
            directory = daf+'/tf_1/ar_'+str(aug_r)
            command = gen_command_best('PointMaze_UMaze-v3', daf=daf, timesteps=300000, aug_r=aug_r, \
                                        subdir=directory)
            print(command)
            f.write(command.replace(' ', '*') + "\n")
            # also add a run without any DA: I'm too lazy to copy and paste 
            # so I'll make a no_DA run for every daf
            directory = daf+'/tf_1/no_DA'
            command = gen_command_best('PointMaze_UMaze-v3', daf=None, timesteps=300000, aug_r=None, subdir=directory)
            print(command)
            f.write(command.replace(' ', '*') + "\n")

    timesteps = 1000000
    for env_id in env_ids[1:]: 
        for daf in pointmaze_dafs:
            for aug_r in aug_ratios:
                directory = daf+'/tf_1/ar_'+str(aug_r)
                command = gen_command_best(env_id, daf=daf, timesteps=timesteps, aug_r=aug_r, \
                                            subdir=directory)
                print(command)
                f.write(command.replace(' ', '*') + "\n")
            
            # run without DA:
            directory = daf+'/tf_1/no_DA'
            command = gen_command_best(env_id, daf=None, timesteps=timesteps, aug_r=None, subdir=directory)
            print(command)
            f.write(command.replace(' ', '*') + "\n")
