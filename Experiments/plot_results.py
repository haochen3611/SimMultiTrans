import SimMultiTrans as smt
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import os.path as ospath
import glob
import json
import numpy as np
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = [
    r'\usepackage{amsmath}',
    r'\usepackage{amssymb}',
    r'\DeclareMathOperator{\Span}{\mathbf{span}}']


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


def plot_performance(res_dict: dict, path: str):
    alpha = res_dict.pop('alpha')
    wait_time = res_dict.pop('wait_time')
    vmt = res_dict.pop('vmt')
    name = f'perf-fs-{res_dict.pop("fleet_size")}.png'
    fig, ax_1 = plt.subplots(constrained_layout=False)
    ax_1.set_xscale('log')
    ax_1.set_yscale('log')
    ax_1.grid()
    ax_2 = ax_1.twinx()
    l_1 = ax_1.plot(alpha, wait_time, 'rd-', label="relative wait time")
    l_2 = ax_2.plot(alpha, vmt, 'bo-', label="relative EVMT")
    ax_1.set_xlabel(r'$\alpha$ - weight on rebalancing cost', fontsize=14)
    ax_1.set_ylabel(r'Relative waiting time',
                    color='r',
                    fontsize=14)
    ax_2.set_ylabel(r'Relative EVMT',
                    color='b',
                    fontsize=14)
    lns = l_1 + l_2
    labs = [_.get_label() for _ in lns]
    ax_1.legend(lns, labs, loc="upper center")
    plt.savefig(ospath.join(path, name))
    plt.close(fig)


def process_lite_results(path):
    f_lst = glob.glob(ospath.join(path, '*.json'))
    avg_wt = 0
    reb_trips = 0
    reb_miles = 0
    avg_miles = 0

    for f in f_lst:
        with open(f, 'r') as file:
            res = json.load(file)
            avg_wt += np.mean(res['avg_wait_time'])
            reb_trips += res['reb_trips']
            reb_miles += res['reb_miles']
            avg_miles += res['reb_miles']/res['reb_trips']

    avg_wt /= len(f_lst)*60
    reb_trips /= len(f_lst)
    reb_miles /= len(f_lst)
    avg_miles /= len(f_lst)

    return {'avg_wait_time': avg_wt,
            'reb_trips': reb_trips,
            'avg_miles': avg_miles,
            'cost_wait': avg_wt*46377,
            'reb_miles': reb_miles}


if __name__ == '__main__':

    POLICY_LOSS = '/home/haochen/PycharmProjects/SimMultiTrans/Experiments/plots/data/loss/run-PPO_TaxiRebalance_2020-05-10_22-55-43rmoartvj_dpr0.24-tag-ray_tune_info_learner_default_policy_policy_loss.csv'
    VF_LOSS = '/home/haochen/PycharmProjects/SimMultiTrans/Experiments/plots/data/loss/run-PPO_TaxiRebalance_2020-05-10_22-55-43rmoartvj_dpr0.24-tag-ray_tune_info_learner_default_policy_vf_loss.csv'
    MEAN_REWARD = '/home/haochen/PycharmProjects/SimMultiTrans/Experiments/plots/data/reward/run-PPO_TaxiRebalance_2020-05-10_22-54-23920b8n7d_dpr0.23-tag-ray_tune_episode_reward_mean.csv'

    DIR = '/home/haochen/PycharmProjects/SimMultiTrans/Experiments/plots/'

    plot_scalar(data=POLICY_LOSS, path=DIR, ylabel='Policy Loss')
    plot_scalar(data=VF_LOSS, path=DIR, ylabel='Value Function Loss')
    plot_scalar(data=MEAN_REWARD, path=DIR, ylabel='Mean Reward per Episode')

    fs_600 = {
        "fleet_size": 600,
        "alpha": [10, 100, 1000],
        "wait_time": [1.79, 3.47, 93.88],
        "vmt": [0.90, 0.92, 0.12]
    }

    fs_1000 = {
        "fleet_size": 1000,
        "alpha": [0.01, 0.1, 1, 10, 100, 1000],
        "wait_time": [0.21, 0.30, 0.35, 0.36, 2.75, 87.47],
        "vmt": [2.14, 1.73, 1.46, 0.98, 0.78, 0.13]
    }

    plot_performance(fs_600, path=DIR)
    plot_performance(fs_1000, path=DIR)

    # e_3 = '/home/haochen/Downloads/Final_Resutls/5/sim_results/home/cc/SimMultiTrans/SimMultiTrans/results/2020-05-30-14-35-31'
    # e_2 = '/home/haochen/Downloads/Final_Resutls/5/sim_results/home/cc/SimMultiTrans/SimMultiTrans/results/2020-05-30-06-23-34'
    # e_1 = '/home/haochen/Downloads/Final_Resutls/4/sim_results/home/cc/SimMultiTrans/SimMultiTrans/results/2020-05-30-14-29-57'
    # e_0 = '/home/haochen/Downloads/Final_Resutls/4/sim_results/home/cc/SimMultiTrans/SimMultiTrans/results/2020-05-30-06-24-22'
    # e_n1 = '/home/haochen/Downloads/Final_Resutls/3/sim_results/home/cc/SimMultiTrans/SimMultiTrans/results/2020-05-30-06-25-01'
    #
    # e_3_res = process_lite_results(e_3)
    # e_2_res = process_lite_results(e_2)
    # e_1_res = process_lite_results(e_1)
    # e_0_res = process_lite_results(e_0)
    # e_n1_res = process_lite_results(e_n1)
    # print(e_n1_res)
    # print(e_0_res)
    # print(e_1_res)
    # print(e_2_res)
    # print(e_3_res)
