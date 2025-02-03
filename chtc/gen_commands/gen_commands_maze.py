import os
'''
Best hyperparams for PointMaze_{UMaze, Medium, Large}-v3, 20 seeds
    Note: extra commandline args since PointMaze has diff params from Panda envs
'''
# python3 ddpg.py --env_id PandaSlide-v3 --daf RelabelGoal --save_subdir 2xTranslateGoal 

# def gen_command_best(env_id, daf, timesteps, learning_rate, batch_size, network, net_subdir):
def gen_command_best(env_id, daf, timesteps, subdir):
    python_command = f'python3 ddpg.py --env_id {env_id} ' \
                    f'--gamma 0.99 ' \
                    f'--exploration_noise 0.1 ' \
                    f'--learning_rate 1e-4 ' \
                    f'--batch_size 64 ' \
                    f'--net_arch 256 256 256 ' \
                    f'--save_subdir {subdir} ' \
                    f'--total_timesteps {timesteps} ' \
                    f'--env_kwargs continuing_task False '
    if daf != None:
        python_command = python_command + f'--daf {daf} ' 
        python_command = python_command + f'--buffer_size {int(2e6)} ' # x2 buffer
    else: # no DAF
        python_command = python_command + f'--buffer_size {int(1e6)} ' # buffer=1mill

    mem = 7.5
    disk = 9.5
    command = f"{mem},{disk},{python_command}"
    return command

if __name__ == "__main__":
    os.makedirs('../commands', exist_ok=True)
    f = open(f"../commands/train_maze.txt", "w")
    
    env_ids = ['PointMaze_UMaze-v3', 'PointMaze_Medium-v3', 'PointMaze_Large-v3']
    pointmaze_dafs = ['RelabelGoal', 'TranslateRotate', 'TranslateRotateRelabelGoal']
    net_archs_param = "256 256 256"
    net_arch_str = "256,256,256"

    # for env in env_ids: 

    # command = gen_command('PointMaze_UMaze-v3', daf=None, timesteps=300000, learning_rate=1e-4, \
    #         batch_size=64, network="256 256 256", subdir="best_params")
    # no daf
    command = gen_command_best('PointMaze_UMaze-v3', daf=None, timesteps=300000, subdir="best_no_DA")
    print(command)
    f.write(command.replace(' ', '*') + "\n")

    command = gen_command_best('PointMaze_Medium-v3', daf=None, timesteps=1000000, subdir="best_no_DA")
    print(command)
    f.write(command.replace(' ', '*') + "\n")

    command = gen_command_best('PointMaze_Large-v3', daf=None, timesteps=1000000, subdir="best_no_DA")
    print(command)
    f.write(command.replace(' ', '*') + "\n")

    #### TODO: later sweep over batch size = 64, 128, 256 again with DA
    for daf in pointmaze_dafs:
        command = gen_command_net(env, lr, batch_size, "256 256 256", "256,256,256")
        print(command)
        f.write(command.replace(' ', '*') + "\n")
    