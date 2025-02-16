import os

from gymnasium.envs.registration import register

ENVS_DIR = os.path.join(os.path.dirname(__file__), 'envs')

############################################################################
### Toy

register(
    id="TwoStep-v0",
    entry_point="custom_envs.envs.two_step:TwoStepEnv",
    max_episode_steps=2,
)

register(
    id="GridWorld-v0",
    entry_point="custom_envs.envs.gridworld:GridWorldEnv",
    max_episode_steps=30,
)

register(
    id="Goal2D-v0",
    entry_point="custom_envs.envs.goal2d:Goal2DEnv",
    max_episode_steps=100,
)

register(
    id="Bandit5-v0",
    entry_point="custom_envs.envs.bandit:BanditEnv",
    max_episode_steps=1,
    kwargs={
        'n': 5
    }
)

register(
    id="Bandit20-v0",
    entry_point="custom_envs.envs.bandit:BanditEnv",
    max_episode_steps=1,
    kwargs={
        'n': 20
    }
)

register(
    id="Bandit100-v0",
    entry_point="custom_envs.envs.bandit:BanditEnv",
    max_episode_steps=1,
    kwargs={
        'n': 100
    }
)

register(
    id="SillyBandit-v0",
    entry_point="custom_envs.envs.bandit:SillyBanditEnv",
    max_episode_steps=1,
    # kwargs={
    #     'n': 10
    # }
)

register(
    id="SillyBanditNegative-v0",
    entry_point="custom_envs.envs.bandit:SillyBanditEnv",
    max_episode_steps=1,
    kwargs={
        'reward': -1
    }
)

register(
    id="Bandit1000-v0",
    entry_point="custom_envs.envs.bandit:BanditEnv",
    max_episode_steps=1,
    kwargs={
        'n': 1000
    }
)

for n in [10, 50, 100]:
    register(
        id=f"Discrete2D{n}-v0",
        entry_point="custom_envs.envs.discrete2d:Discrete2DEnv",
        max_episode_steps=50,
        kwargs={
            'n': n
        }
    )

register(
    id="GridWorldHard-5x5-v0",
    entry_point="custom_envs.envs:GridWorldHardEnv",
    max_episode_steps=8,
    kwargs={
        'shape': (5, 5),
    },
)

for l in [3, 4, 5]:
    register(
        id=f"GridWorldCliff-{l}x{2*l}-v0",
        entry_point="custom_envs.envs:GridWorldCliffEnv",
        max_episode_steps=(2*l+l)*2,
        kwargs={
            'shape': (l, 2*l),
        },
    )

for l in [3, 5, 7, 9, 11, 10, 20]:
    register(
        id=f"GridWorld-{l}x{l}-v0",
        entry_point="custom_envs.envs:GridWorldEnv",
        max_episode_steps=3*l,
        kwargs={
            'shape': (l, l),
        },
    )

    register(
        id=f"GridWorldContinuing-{l}x{l}-v0",
        entry_point="custom_envs.envs:GridWorldEnv",
        max_episode_steps=100000,
        kwargs={
            'shape': (l, l),
        },
    )

    register(
        id=f"Chain-{l}-v0",
        entry_point="custom_envs.envs.chain:ChainEnv",
        max_episode_steps=l,
        kwargs={
            'n': l
        }
    )

