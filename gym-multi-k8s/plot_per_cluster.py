import glob
import logging
import os
from collections import namedtuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def get_statistics(df, alg_name, avg_reward, ci_avg_reward, avg_latency, ci_avg_latency,
                   avg_cost, ci_avg_cost, avg_ep_block_prob, ci_avg_ep_block_prob, avg_executionTime):
    '''
    print("{} reward Mean: {}".format(alg_name, np.mean(df["reward"])))
    print("{} reward Std: {}".format(alg_name, np.std(df["reward"])))

    print("{} latency Mean: {}".format(alg_name, np.mean(df["avg_latency"])))
    print("{} latency Std: {}".format(alg_name, np.std(df["avg_latency"])))

    print("{} cost Mean: {}".format(alg_name, np.mean(df["avg_cost"])))
    print("{} cost Std: {}".format(alg_name, np.std(df["avg_cost"])))

    print("{} ep block prob Mean: {}".format(alg_name, np.mean(df["ep_block_prob"])))
    print("{} ep block prob Std: {}".format(alg_name, np.std(df["ep_block_prob"])))

    print("{} executionTime Mean: {}".format(alg_name, np.mean(df["executionTime"])))
    print("{} executionTime Std: {}".format(alg_name, np.std(df["executionTime"])))
    '''
    avg_reward.append(np.mean(df["reward"]))
    ci_avg_reward.append(np.std(df["reward"]))
    avg_latency.append(np.mean(df["avg_latency"]))
    ci_avg_latency.append(np.std(df["avg_latency"]))
    avg_cost.append(np.mean(df["avg_cost"]))
    ci_avg_cost.append(np.std(df["avg_cost"]))
    avg_ep_block_prob.append(np.mean(df["ep_block_prob"]))
    ci_avg_ep_block_prob.append(np.std(df["ep_block_prob"]))
    avg_executionTime.append(np.mean(df["executionTime"]))


if __name__ == "__main__":
    reward = 'latency'  # cost, risk or latency
    max_reward = 100  # cost= 1500, risk and latency= 100
    ylim = 120  # 1700 for cost and 120 for rest

    # testing
    path_ppo = "results/testing/run_2/" + reward + "/ppo_deepsets/"
    path_dqn = "results/testing/run_2/" + reward + "/dqn_deepsets/"

    avg_reward_ppo = []
    ci_avg_reward_ppo = []
    avg_latency_ppo = []
    ci_avg_latency_ppo = []
    avg_cost_ppo = []
    ci_avg_cost_ppo = []
    avg_ep_block_prob_ppo = []
    ci_avg_ep_block_prob_ppo = []
    avg_executionTime_ppo = []

    avg_reward_dqn = []
    ci_avg_reward_dqn = []
    avg_latency_dqn = []
    ci_avg_latency_dqn = []
    avg_cost_dqn = []
    ci_avg_cost_dqn = []
    avg_ep_block_prob_dqn = []
    ci_avg_ep_block_prob_dqn = []
    avg_executionTime_dqn = []

    if os.path.exists(path_ppo):
        for file in glob.glob(f"{path_ppo}/*_gym_*.csv"):
            print(f"\n######### Opening {file} #########")
            df = pd.read_csv(file)
            get_statistics(df, file,
                           avg_reward_ppo, ci_avg_reward_ppo,
                           avg_latency_ppo, ci_avg_latency_ppo,
                           avg_cost_ppo, ci_avg_cost_ppo,
                           avg_ep_block_prob_ppo, ci_avg_ep_block_prob_ppo,
                           avg_executionTime_ppo)

    if os.path.exists(path_dqn):
        for file in glob.glob(f"{path_dqn}/*_gym_*.csv"):
            print(f"\n######### Opening {file} #########")
            df = pd.read_csv(file)
            get_statistics(df, file,
                           avg_reward_dqn, ci_avg_reward_dqn,
                           avg_latency_dqn, ci_avg_latency_dqn,
                           avg_cost_dqn, ci_avg_cost_dqn,
                           avg_ep_block_prob_dqn, ci_avg_ep_block_prob_dqn,
                           avg_executionTime_dqn)

    # Accumulated Reward
    fig = plt.figure()
    x = [4, 8, 12, 16, 25, 32, 48, 64, 80, 128]

    plt.errorbar(x, avg_reward_ppo, yerr=ci_avg_reward_ppo, linestyle=None,
                 marker="s", color='#3399FF', label='Deepsets PPO', markersize=6)

    plt.errorbar(x, avg_reward_dqn, yerr=ci_avg_reward_dqn, color='#EDB120',
                 linestyle='dotted', marker="s", label='Deepsets DQN', markersize=6)

    # specifying horizontal line type
    plt.axhline(y=max_reward, color='black', linestyle='--', label="max reward= " + str(max_reward))
    # plt.yscale('log')

    # set x and y limits
    plt.xlim(0, 129)
    plt.ylim(0, ylim)

    # set x-axis label
    plt.xlabel("Total Number of clusters", fontsize=14)

    # set y-axis label
    plt.ylabel("Accumulated Reward", fontsize=14)

    # show and save figure
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig('plot_per_cluster_reward.png', dpi=250, bbox_inches='tight')
    plt.close()

    # Avg. Cost
    plt.errorbar(x, avg_cost_ppo, yerr=ci_avg_cost_ppo, linestyle=None, marker="s", color='#3399FF',
                 label='Deepsets PPO', markersize=6)

    plt.errorbar(x, avg_cost_dqn, yerr=ci_avg_cost_dqn, color='#EDB120',
                 linestyle='dotted', marker="s", label='Deepsets DQN', markersize=6)

    # specifying horizontal line type
    # plt.axhline(y=1800, color='black', linestyle='--', label="max reward= 1500 ")
    # plt.yscale('log')

    # set x and y limits
    plt.xlim(0, 129)
    plt.ylim(0, 18)

    # set x-axis label
    plt.xlabel("Total Number of clusters", fontsize=14)

    # set y-axis label
    plt.ylabel("Deployment Cost", fontsize=14)

    # show and save figure
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig('plot_per_cluster_cost.png', dpi=250, bbox_inches='tight')
    plt.close()

    # Avg latency
    plt.errorbar(x, avg_latency_ppo, yerr=ci_avg_latency_ppo, linestyle=None, marker="s", color='#3399FF',
                 label='Deepsets PPO', markersize=6)

    plt.errorbar(x, avg_latency_dqn, yerr=ci_avg_latency_dqn, color='#EDB120',
                 linestyle='dotted', marker="s", label='Deepsets DQN', markersize=6)

    # specifying horizontal line type
    # plt.axhline(y=1800, color='black', linestyle='--', label="max reward= 1500 ")
    # plt.yscale('log')

    # set x and y limits
    plt.xlim(0, 129)
    plt.ylim(0, 500)

    # set x-axis label
    plt.xlabel("Total Number of clusters", fontsize=14)

    # set y-axis label
    plt.ylabel("Avg. Latency (in ms)", fontsize=14)

    # show and save figure
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig('plot_per_cluster_latency.png', dpi=250, bbox_inches='tight')
    plt.close()

    # Episode Block Prob
    plt.errorbar(x, avg_ep_block_prob_ppo, yerr=ci_avg_ep_block_prob_ppo, linestyle=None, marker="s", color='#3399FF',
                 label='Deepsets PPO',
                 markersize=6)
    plt.errorbar(x, avg_ep_block_prob_dqn, yerr=ci_avg_ep_block_prob_dqn, color='#EDB120',
                 linestyle='dotted', marker="s", label='Deepsets DQN', markersize=6)

    # specifying horizontal line type
    # plt.axhline(y=1800, color='black', linestyle='--', label="max reward= 1500 ")
    # plt.yscale('log')

    # set x and y limits
    plt.xlim(0, 129)
    plt.ylim(0, 0.15)

    # set x-axis label
    plt.xlabel("Total Number of clusters", fontsize=14)

    # set y-axis label
    plt.ylabel("Percentage of Rejected Requests", fontsize=14)

    # show and save figure
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig('plot_per_cluster_rejected_requests.png', dpi=250, bbox_inches='tight')
    plt.close()
