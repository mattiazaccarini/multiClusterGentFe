import glob
import logging
import os
from collections import namedtuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def get_statistics(df, alg_name, avg_reward, ci_avg_reward, avg_latency, avg_cost, ci_avg_cost, avg_executionTime):
    print("{} reward Mean: {}".format(alg_name, np.mean(df["reward"])))
    print("{} reward Std: {}".format(alg_name, np.std(df["reward"])))

    print("{} latency Mean: {}".format(alg_name, np.mean(df["avg_latency"])))
    print("{} latency Std: {}".format(alg_name, np.std(df["avg_latency"])))

    print("{} cost Mean: {}".format(alg_name, np.mean(df["avg_cost"])))
    print("{} cost Std: {}".format(alg_name, np.std(df["avg_cost"])))

    print("{} executionTime Mean: {}".format(alg_name, np.mean(df["executionTime"])))
    print("{} executionTime Std: {}".format(alg_name, np.std(df["executionTime"])))

    avg_reward.append(np.mean(df["reward"]))
    ci_avg_reward.append(np.std(df["reward"]))
    avg_latency.append(np.mean(df["avg_latency"]))
    avg_cost.append(np.mean(df["avg_cost"]))
    ci_avg_cost.append(np.std(df["avg_cost"]))
    avg_executionTime.append(np.mean(df["executionTime"]))


if __name__ == "__main__":
    reward = 'cost'  # cost, risk or latency

    # testing
    path = "results/testing/" + reward + "/ppo_deepsets/"
    avg_reward = []
    ci_avg_reward = []
    avg_latency = []
    avg_cost = []
    ci_avg_cost = []
    avg_executionTime = []

    if os.path.exists(path):
        for file in glob.glob(f"{path}/*_gym_*.csv"):
            print(f"\n######### Opening {file} #########")
            df = pd.read_csv(file)
            get_statistics(df, file, avg_reward, ci_avg_reward, avg_latency, avg_cost, ci_avg_cost, avg_executionTime)

    fig = plt.figure()
    x = [4, 8, 12, 16, 32, 64, 128]

    plt.errorbar(x, avg_reward, yerr=ci_avg_reward, linestyle=None, marker="s", label='Deepsets PPO',
                 markersize=6)
    #plt.errorbar(x, qos_sort, yerr=ci_qos_sort, color='#EDB120', linestyle='dotted', marker="s", label='QoS Sort',
                 #markersize=6)

    # specifying horizontal line type
    plt.axhline(y=1500, color='black', linestyle='--', label="max reward= 1500 ")
    # plt.yscale('log')

    # set x and y limits
    plt.xlim(0, 129)
    plt.ylim(0, 1800)

    # set x-axis label
    plt.xlabel("Total Number of clusters", fontsize=14)

    # set y-axis label
    plt.ylabel("Accumulated Reward", fontsize=14)

    # show and save figure
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig('reward.png', dpi=250, bbox_inches='tight')
    plt.close()

    plt.errorbar(x, avg_cost, yerr=ci_avg_cost, linestyle=None, marker="s", label='Deepsets PPO',
                 markersize=6)
    # plt.errorbar(x, qos_sort, yerr=ci_qos_sort, color='#EDB120', linestyle='dotted', marker="s", label='QoS Sort',
    # markersize=6)

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
    plt.savefig('cost.png', dpi=250, bbox_inches='tight')
    plt.close()
