import os
'''
Recreating Panda envs in ICLR Fig.3 
    1xPolicy:  buffer_size 1e6
    2xViaTranslateGoal: buffer_size 2e6
    2xPolicy: buffer_size 2e6, 2x batch, 2x total_timesteps 
'''
# python3 ddpg.py --env_id PandaSlide-v3 --daf RelabelGoal --save_subdir 2xTranslateGoal 

def gen_command(env_id, daf, batch_size, subdir, timesteps):
    python_command = f'python3 ddpg.py --env_id {env_id} ' \
                    f'--batch_size {batch_size} ' \
                    f'--save_subdir {subdir} '  \
                    f'--gamma 0.95 ' \

    if daf != None:
        python_command = python_command + f'--daf {daf} ' 
        python_command = python_command + f'--buffer_size {int(2e6)} ' # x2 buffer
    else: # no DAF
        python_command = python_command + f'--buffer_size {int(1e6)} ' # buffer=1M

    if timesteps != None:
        python_command = python_command + f'--total_timesteps {int(timesteps)}'

    mem = 9
    disk = 10
    command = f"{mem},{disk},{python_command}"
    return command

def gen_command_more(env_id, daf, batch_size, subdir, buffer_size, train_freq, timesteps, eval_freq):
    python_command = f'python3 ddpg.py --env_id {env_id} ' \
                    f'--batch_size {batch_size} ' \
                    f'--train_freq {train_freq} ' \
                    f'--eval_freq {eval_freq} ' \
                    f'--save_subdir {subdir} ' \
                    f'--buffer_size {int(buffer_size)} ' \
                    f'--gamma 0.95 ' \

    if timesteps != None:
        python_command = python_command + f'--total_timesteps {int(timesteps)} ' 
    if daf != None:
        python_command = python_command + f'--daf {daf} '

    mem = 9
    disk = 10
    command = f"{mem},{disk},{python_command}"
    return command

def gen_command_aug_r(env_id, daf, batch_size, subdir, buffer_size, aug_buffer_size, train_freq, timesteps, eval_freq, aug_r):
    python_command = f'python3 ddpg.py --env_id {env_id} ' \
                    f'--learning_rate 0.001 ' \
                    f'--noise_clip 0.5 ' \
                    f'--random_action_prob 0.3 ' \
                    f'--aug_ratio {aug_r} ' \
                    f'--batch_size {batch_size} ' \
                    f'--train_freq {train_freq} ' \
                    f'--eval_freq {eval_freq} ' \
                    f'--save_subdir {subdir} ' \
                    f'--buffer_size {int(buffer_size)} ' \
                    f'--gamma 0.95 ' \

    if timesteps != None:
        python_command = python_command + f'--total_timesteps {int(timesteps)} ' 
    if daf != None:
        python_command = python_command + f'--daf {daf} '
        python_command = python_command + f'--aug_buffer_size {int(aug_buffer_size)}'
 
 # python3 ddpg.py --env_id PandaPush-v3 --batch_size 1024 --train_freq 4 
 # --eval_freq 20000 --save_subdir 2xPolicy 
 # --buffer_size 1000000 --gamma 0.95 --total_timesteps 2000
    mem = 9
    disk = 10
    command = f"{mem},{disk},{python_command}"
    return command

if __name__ == "__main__":
    os.makedirs('commands', exist_ok=True)
    f = open(f"commands/train_panda_aug_r.txt", "w")

    env_ids = ['PandaPush-v3', 'PandaSlide-v3']
    daf = 'RelabelGoal'
    aug_ratios = [1,2,4,8]
    for env in env_ids:
        # command = gen_command(env, daf=None, batch_size=256, subdir='1xPolicy', timesteps=None) # no DA
        # print(command)
        # f.write(command.replace(' ', '*') + "\n")
        
        # command = gen_command(env, daf=daf, batch_size=256*2, subdir='2xViaTranslateGoal', timesteps=None) # 
        # print(command)
        # f.write(command.replace(' ', '*') + "\n")

        # 2xPolicy -> 2x buffer, batch, and total_timesteps
        # command = gen_command_more(env, daf=None, batch_size=256*2, subdir='2xPolicy', \
        #                                 buffer_size=2e6, train_freq=4, timesteps=2e6, eval_freq=20000) 
        # print(command)
        # f.write(command.replace(' ', '*') + "\n")

        # With DA: aug_r sweep
        for aug_r in aug_ratios:
            command = gen_command_aug_r(env, daf=daf, batch_size=256*2, subdir='TranslateGoal', \
                                        buffer_size=1000000, aug_buffer_size=1000000, train_freq=2, \
                                        timesteps=1000000, eval_freq=20000, aug_r=aug_r)
            print(command)
            f.write(command.replace(' ', '*') + "\n")


    # env_ids = ['PandaPickAndPlace-v3', 'PandaFlip-v3']
    env_ids = ['PandaPickAndPlace-v3']
    for env in env_ids:
        # command = gen_command(env, daf=None, batch_size=512, subdir='1xPolicy', timesteps=1.5e6) 
        # print(command)
        # f.write(command.replace(' ', '*') + "\n")
        
        # command = gen_command(env, daf, batch_size=512*2, subdir='2xViaTranslateGoal', timesteps=1.5e6) 
        # print(command)
        # f.write(command.replace(' ', '*') + "\n")

        # 2xPolicy -> 2x buffer, batch, and total_timesteps
        # command = gen_command_more(env, daf=None, batch_size=512*2, subdir='2xPolicy', \
        #                             buffer_size=2e6, train_freq=4, timesteps=3e6, eval_freq=20000)  
        # print(command)
        # f.write(command.replace(' ', '*') + "\n")

        for aug_r in aug_ratios:
            command = gen_command_aug_r(env, daf=daf, batch_size=512*2, subdir='TranslateGoal', \
                                        buffer_size=1000000, aug_buffer_size=1000000, train_freq=2, \
                                        timesteps=1.5e6, eval_freq=20000, aug_r=aug_r)
            print(command)
            f.write(command.replace(' ', '*') + "\n")
