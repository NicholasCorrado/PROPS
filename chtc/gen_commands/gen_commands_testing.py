import os
'''
Gen commands for testing chtc jobs
'''
# python3 ddpg.py --env_id PandaSlide-v3 --daf RelabelGoal --save_subdir 2xTranslateGoal 

# def gen_command_best(env_id, daf, timesteps, learning_rate, batch_size, network, net_subdir):
def gen_command(env_id):
    python_command = f'python3 ddpg.py --env_id {env_id} --total_timesteps 50000' 

    mem = 5
    disk = 10
    command = f"{mem},{disk},{python_command}"
    return command

if __name__ == "__main__":
    os.makedirs('../commands', exist_ok=True)
    f = open(f"../commands/testing.txt", "w")

    env_ids = ['PandaPush-v3', 'PandaSlide-v3']
    for env in env_ids:
        command = gen_command(env) # no DA
        print(command)
        f.write(command.replace(' ', '*') + "\n")
