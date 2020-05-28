import SimMultiTrans as smt
import matplotlib.pyplot as plt
import pandas as pd
import os.path as ospath


def plot_metric():
    graph = smt.default_graph()
    plt = smt.Plot(graph=graph)
    plt.import_results()

    plt.plot_passenger_queuelen_time(mode='taxi')
    plt.plot_passenger_waittime(mode='taxi')
    plt.plot_metrics(mode='taxi')


def plot_scalar(data, path, **kwargs):
    if isinstance(data, str):
        df = pd.read_csv(data, index_col=False)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise TypeError
    title = kwargs.pop('title', '')
    xlabel = kwargs.pop('xlabel', 'Steps')
    ylabel = kwargs.pop('ylabel', '')
    if title:
        name = title
    elif ylabel:
        name = ylabel
    else:
        raise ValueError
    df.plot(x='Step', y='Value', legend=False, fontsize=14)
    plt.title(title)
    plt.xticks((0, 5e6, 10e6, 15e6, 20e6, 25e6), ('0', '5M', '10M', '15M', '20M', '25M'))
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.tight_layout()
    plt.grid()
    plt.savefig(ospath.join(path, f'{"-".join(name.split(" "))}.png'))
    plt.close()


if __name__ == '__main__':

    POLICY_LOSS = '/home/haochen/PycharmProjects/SimMultiTrans/Experiments/plots/data/loss/run-PPO_TaxiRebalance_2020-05-10_22-55-43rmoartvj_dpr0.24-tag-ray_tune_info_learner_default_policy_policy_loss.csv'
    VF_LOSS = '/home/haochen/PycharmProjects/SimMultiTrans/Experiments/plots/data/loss/run-PPO_TaxiRebalance_2020-05-10_22-55-43rmoartvj_dpr0.24-tag-ray_tune_info_learner_default_policy_vf_loss.csv'
    MEAN_REWARD = '/home/haochen/PycharmProjects/SimMultiTrans/Experiments/plots/data/reward/run-PPO_TaxiRebalance_2020-05-10_22-54-23920b8n7d_dpr0.23-tag-ray_tune_episode_reward_mean.csv'

    DIR = '/home/haochen/PycharmProjects/SimMultiTrans/Experiments/plots/'

    plot_scalar(data=POLICY_LOSS, path=DIR, ylabel='Policy Loss')
    plot_scalar(data=VF_LOSS, path=DIR, ylabel='Value Function Loss')
    plot_scalar(data=MEAN_REWARD, path=DIR, ylabel='Mean Reward per Episode')

