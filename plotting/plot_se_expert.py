import os

import numpy as np
import seaborn

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from rliable import library as rly
from rliable import metrics
# from rliable.plot_utils import plot_sample_efficiency_curve

from utils import get_data, plot_sample_efficiency_curve

KEY_MAPPING = {
    'props': 'PROPS',
    'ros': 'ROS',
    'on_policy': 'On-Policy Sampling',
}

if __name__ == "__main__":

    seaborn.set_theme(style='whitegrid')

    n_rows = 1
    n_cols = 3
    fig = plt.figure(figsize=(n_cols*6,n_rows*6))
    i = 1

    # env_ids = ['Swimmer-v5', 'Hopper-v5', 'HalfCheetah-v5', 'Walker2d-v5', 'Ant-v5', 'Humanoid-v5']
    env_ids = [
        'CartPole-v1',
        'LunarLander-v3',
        'Discrete2D100-v0',
    ]

    for env_id in env_ids:
        ax = plt.subplot(n_rows, n_cols, i)
        i+=1
        x_dict = {}
        data_dict = {}

        # for sampling_algo in ['props', 'ros', 'on_policy']:
        # for sampling_algo in ['on_policy']:

        sampling_algo = 'on_policy'
        key = KEY_MAPPING[sampling_algo]
        results_dir = f"../chtc/results/se_discrete_expert/results/{env_id}/ppo/{sampling_algo}/"
        x, y = get_data(results_dir, x_name='timestep', y_name='sampling_error', filename='evaluations.npz')
        if y is not None:
            x_dict[key] = x
            data_dict[key] = y
        #
        # sampling_algo = 'props'
        # key = KEY_MAPPING[sampling_algo]
        # results_dir = f"../chtc/results/se_discrete_expert_props/results/{env_id}/ppo/{sampling_algo}"
        # x, y = get_data(results_dir, x_name='timestep', y_name='sampling_error', filename='evaluations.npz')
        # if y is not None:
        #     x_dict[key] = x
        #     data_dict[key] = y

        sampling_algo = 'ros'
        # key = KEY_MAPPING[sampling_algo] + f" plr_{plr}/pns_{pns}/pkl_{pkl}"
        key = KEY_MAPPING[sampling_algo]
        results_dir = f"../chtc/results/se_discrete_expert/results/{env_id}/ppo/{sampling_algo}"
        x, y = get_data(results_dir, x_name='timestep', y_name='sampling_error', filename='evaluations.npz')
        if y is not None:
            x_dict[key] = x
            data_dict[key] = y


        for plr in [1e-4]:
            for pns in [16]:
                for pkl in [0.01]:
                    for pe in [32]:
                        for pc in [0.3]:
                            sampling_algo = 'props'
                            key = KEY_MAPPING[sampling_algo] + f" plr_{plr}/pe_{pe}/pns_{pns}/pc_{pc}/pkl_{pkl}"
                            # key = KEY_MAPPING[sampling_algo] + f" behavior batch size = {pns}"
                            results_dir = f"../local/tmp/{env_id}/ppo/{sampling_algo}/plr_{plr}/pe_{pe}/pns_{pns}/pc_{pc}/pkl_{pkl}"
                            x, y = get_data(results_dir, x_name='timestep', y_name='sampling_error', filename='evaluations.npz')
                            if y is not None:
                                x_dict[key] = x
                                data_dict[key] = y

        results_dict = {algorithm: score for algorithm, score in data_dict.items()}
        aggr_func = lambda scores: np.array([metrics.aggregate_mean([scores[..., frame]]) for frame in range(scores.shape[-1])])
        scores, cis = rly.get_interval_estimates(results_dict, aggr_func, reps=100)

        plot_sample_efficiency_curve(
            frames=x_dict,
            point_estimates=scores,
            interval_estimates=cis,
            ax=ax,
            algorithms=None,
            xlabel='Timestep',
            ylabel=f'Sampling Error',
            # title=f'{env_id}',
            labelsize='large',
            ticklabelsize='large',
            marker=None
        )
        # plt.xticks(ticks=[256, 512, 1024, 2048, 4096], labels=[256, 512, 1024, 2048, 4096])
        # Set the x-ticks locations
        ax.set_xticks([256, 512, 1024, 2048, 4096])

        # Set the x-ticks labels
        ax.set_xticklabels([256, 512, 1024, 2048, 4096])
        ax.set_title(f'{env_id}', fontsize='large')
        # Use scientific notation for x-axis
        # plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        # set fontsize of scientific notation label
        ax.xaxis.get_offset_text().set_fontsize('large')

        # Set log scale
        plt.xscale('log')
        plt.yscale('log')

        plt.tight_layout()

    # Push plots down to make room for the the legend
    fig.subplots_adjust(top=0.8)

    # Fetch and plot the legend from one of the subplots.
    ax = fig.axes[0]
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', fontsize='large', ncols=3)

    save_dir = f'figures'
    save_name = f'sampling_error.png'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/{save_name}')

    plt.show()
