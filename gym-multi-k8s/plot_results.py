import logging
from collections import namedtuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

stats = namedtuple("episode_stats", ["a2c_rewards", "mask_ppo_rewards", "deepsets_rewards",
                                     "a2c_ep_block_prob", "mask_ppo_ep_block_prob", "deepsets_ep_block_prob",
                                     "a2c_latency", "mask_ppo_latency", "deepsets_latency",
                                     ])



def plot_stats(figName, stats, smoothing_window=10):
    # Plot the episode reward over time
    a2c_rewards = pd.Series(stats.a2c_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()

    mask_ppo_rewards = pd.Series(stats.mask_ppo_rewards).rolling(smoothing_window,
                                                                 min_periods=smoothing_window).mean()

    deepsets_rewards = pd.Series(stats.deepsets_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()

    a2c_ep_block_prob = pd.Series(stats.a2c_ep_block_prob).rolling(smoothing_window, min_periods=smoothing_window).mean()
    mask_ppo_ep_block_prob = pd.Series(stats.mask_ppo_ep_block_prob).rolling(smoothing_window,
                                                                 min_periods=smoothing_window).mean()
    deepsets_ep_block_prob = pd.Series(stats.deepsets_ep_block_prob).rolling(smoothing_window, min_periods=smoothing_window).mean()

    a2c_latency = pd.Series(stats.a2c_latency).rolling(smoothing_window,
                                                                   min_periods=smoothing_window).mean()
    mask_ppo_latency = pd.Series(stats.mask_ppo_latency).rolling(smoothing_window,
                                                                             min_periods=smoothing_window).mean()
    deepsets_latency = pd.Series(stats.deepsets_latency).rolling(smoothing_window,
                                                   min_periods=smoothing_window).mean()

    fig = plt.figure()
    plt.plot(a2c_rewards, label='A2C')
    # plt.plot(ppo_sim_rewards, label='PPO (Simulation)')
    plt.plot(mask_ppo_rewards, label='Maskable PPO')
    # plt.plot(a2c_rewards, label='A2C (Cluster)')
    # plt.plot(ppo_rewards, label='PPO (Cluster)')
    plt.plot(deepsets_rewards, label='Deepsets PPO')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.xlim(100, 1500)
    plt.ylim(0, 100)
    plt.legend()
    # plt.title("Episode Reward (Smoothed over window size {})".format(smoothing_window))
    plt.savefig(figName + '_reward.png', dpi=250, bbox_inches='tight')

    fig = plt.figure()
    plt.plot(a2c_ep_block_prob, label='A2C')
    # plt.plot(ppo_sim_rewards, label='PPO (Simulation)')
    plt.plot(mask_ppo_ep_block_prob, label='Maskable PPO')
    # plt.plot(a2c_rewards, label='A2C (Cluster)')
    # plt.plot(ppo_rewards, label='PPO (Cluster)')
    plt.plot(deepsets_ep_block_prob, label='Deepsets PPO')
    plt.xlabel("Episode")
    plt.ylabel("Percentage of Rejected Requests")
    plt.xlim(100, 1500)
    plt.ylim(0, 1)
    plt.legend()
    # plt.title("Episode Reward (Smoothed over window size {})".format(smoothing_window))
    plt.savefig(figName + '_block_probability.png', dpi=250, bbox_inches='tight')

    fig = plt.figure()
    plt.plot(a2c_latency, label='A2C')
    # plt.plot(ppo_sim_rewards, label='PPO (Simulation)')
    plt.plot(mask_ppo_latency, label='Maskable PPO')
    # plt.plot(a2c_rewards, label='A2C (Cluster)')
    # plt.plot(ppo_rewards, label='PPO (Cluster)')
    plt.plot(deepsets_latency, label='Deepsets PPO')
    plt.xlabel("Episode")
    plt.ylabel("Avg. Latency (in ms)")
    plt.xlim(100, 1500)
    plt.ylim(0, 300)
    plt.legend()
    # plt.title("Episode Reward (Smoothed over window size {})".format(smoothing_window))
    plt.savefig(figName + '_latency.png', dpi=250, bbox_inches='tight')

    '''
    fig = plt.figure()
    plt.plot(a2c_sim_cost, label='A2C (Simulation)')
    # plt.plot(ppo_sim_cost, label='PPO (Simulation)')
    plt.plot(recurrent_ppo_sim_cost, label='RPPO (Simulation)')
    plt.plot(a2c_cost, label='A2C (Cluster)')
    # plt.plot(ppo_cost, label='PPO (Cluster)')
    plt.plot(recurrent_ppo_cost, label='RPPO (Cluster)')
    plt.xlabel("Episode")
    plt.ylabel("Number of deployed Pods")
    plt.xlim(100, 2020)
    plt.legend()
    plt.title("Deployment Cost (Smoothed over window size {})".format(smoothing_window))
    plt.savefig(figName + '_cost.png', dpi=250, bbox_inches='tight')

    fig = plt.figure()
    plt.plot(a2c_sim_latency, label='A2C (Simulation)')
    # plt.plot(ppo_sim_latency, label='PPO (Simulation)')
    plt.plot(recurrent_ppo_sim_latency, label='RPPO (Simulation)')
    plt.plot(a2c_latency, label='A2C (Cluster)')
    # plt.plot(ppo_latency, label='PPO (Cluster)')
    plt.plot(recurrent_ppo_latency, label='RPPO (Cluster)')
    plt.xlabel("Episode")
    plt.ylabel("Latency (in ms)")
    plt.xlim(100, 2020)
    plt.legend()
    plt.title("Latency (Smoothed over window size {})".format(smoothing_window))
    plt.savefig(figName + '_latency.png', dpi=250, bbox_inches='tight')
    '''


def remove_empty_lines(df):
    print(df.isnull().sum())
    # Droping the empty rows
    modified = df.dropna()
    # Saving it to the csv file
    modified.to_csv('karmada_gym_results.csv', index=False)
    return modified


if __name__ == "__main__":
    file_a2c = "results/karmada/risk/" \
               "a2c_env_karmada_num_clusters_4_reward_risk_totalSteps_200000_run_1" \
               "/karmada_gym_results.csv"
    file_mask_ppo = "results/karmada/risk/" \
                    "mask_ppo_env_karmada_num_clusters_4_reward_risk_totalSteps_200000_run_1" \
                    "/karmada_gym_results.csv"
    file_deepsets_ppo = "results/karmada/risk/" \
                        "ppo_deepsets_env_karmada_num_clusters_4_reward_risk_totalSteps_500000_run_1" \
                        "/karmada_gym_results.csv"

    # df_ppo = pd.read_csv(file_ppo)
    df_a2c = pd.read_csv(file_a2c)
    df_mask_ppo = pd.read_csv(file_mask_ppo)
    df_deepsets_ppo = pd.read_csv(file_deepsets_ppo)

    # remove_empty_lines(df_a2c)
    # remove_empty_lines(df_mask_ppo)
    # remove_empty_lines(df_deepsets_ppo)

    stats = stats(
        a2c_rewards=df_a2c['reward'],
        mask_ppo_rewards=df_mask_ppo['reward'],
        deepsets_rewards=df_deepsets_ppo['reward'],
        a2c_ep_block_prob=df_a2c['ep_block_prob'],
        mask_ppo_ep_block_prob=df_mask_ppo['ep_block_prob'],
        deepsets_ep_block_prob=df_deepsets_ppo['ep_block_prob'],
        a2c_latency=df_a2c['avg_latency'],
        mask_ppo_latency=df_mask_ppo['avg_latency'],
        deepsets_latency=df_deepsets_ppo['avg_latency'],
    )

    plot_stats("karmada_risk", stats, 100)

    '''
    file = "testing/onlineboutique/latency/real/rppo/results.csv"
    df = pd.read_csv(file)
    # print(df.head)
    # hist = df["cpu"].hist(bins=4)
    # plt.show()
    '''

    df = df_a2c
    print("a2c reward Mean: " + str(np.mean(df["reward"])))
    print("a2c reward Std: " + str(np.std(df["reward"])))

    print("a2c avg_latency Mean: " + str(np.mean(df["avg_latency"])))
    print("a2c avg_latency Std: " + str(np.std(df["avg_latency"])))

    print("a2c execution_time Mean: " + str(np.mean(df["executionTime"])))
    print("a2c execution_time Std: " + str(np.std(df["executionTime"])))

    df = df_mask_ppo
    print("mask ppo reward Mean: " + str(np.mean(df["reward"])))
    print("mask ppo reward Std: " + str(np.std(df["reward"])))

    print("mask ppo avg_latency Mean: " + str(np.mean(df["avg_latency"])))
    print("mask ppo avg_latency Std: " + str(np.std(df["avg_latency"])))

    print("mask ppo execution_time Mean: " + str(np.mean(df["executionTime"])))
    print("mask ppo execution_time Std: " + str(np.std(df["executionTime"])))

    df = df_deepsets_ppo
    print("deepsets_ppo reward Mean: " + str(np.mean(df["reward"])))
    print("deepsets_ppo reward Std: " + str(np.std(df["reward"])))

    print("deepsets_ppo avg_latency Mean: " + str(np.mean(df["avg_latency"])))
    print("deepsets_ppo avg_latency Std: " + str(np.std(df["avg_latency"])))

    print("deepsets_ppo execution_time Mean: " + str(np.mean(df["executionTime"])))
    print("deepsets_ppo execution_time Std: " + str(np.std(df["executionTime"])))
