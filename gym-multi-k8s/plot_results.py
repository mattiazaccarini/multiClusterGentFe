import logging
from collections import namedtuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

stats = namedtuple("episode_stats", ["a2c_rewards", "mask_ppo_rewards", "deepsets_rewards", "deepsets_dqn_rewards",
                                     "a2c_ep_block_prob", "mask_ppo_ep_block_prob", "deepsets_ep_block_prob",
                                     "deepsets_dqn_ep_block_prob",
                                     "a2c_latency", "mask_ppo_latency", "deepsets_latency", "deepsets_dqn_latency",
                                     "a2c_cost", "mask_ppo_cost", "deepsets_cost", "deepsets_dqn_cost"
                                     ])


def plot_stats(figName, stats, smoothing_window=10):
    # Plot the episode reward over time
    a2c_rewards = pd.Series(stats.a2c_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()

    mask_ppo_rewards = pd.Series(stats.mask_ppo_rewards).rolling(smoothing_window,
                                                                 min_periods=smoothing_window).mean()

    deepsets_rewards = pd.Series(stats.deepsets_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()

    deepsets_dqn_rewards = pd.Series(stats.deepsets_dqn_rewards).rolling(smoothing_window,
                                                                         min_periods=smoothing_window).mean()

    a2c_ep_block_prob = pd.Series(stats.a2c_ep_block_prob).rolling(smoothing_window,
                                                                   min_periods=smoothing_window).mean()
    mask_ppo_ep_block_prob = pd.Series(stats.mask_ppo_ep_block_prob).rolling(smoothing_window,
                                                                             min_periods=smoothing_window).mean()
    deepsets_ep_block_prob = pd.Series(stats.deepsets_ep_block_prob).rolling(smoothing_window,
                                                                             min_periods=smoothing_window).mean()

    deepsets_dqn_ep_block_prob = pd.Series(stats.deepsets_dqn_ep_block_prob).rolling(smoothing_window,
                                                                                     min_periods=smoothing_window).mean()

    a2c_latency = pd.Series(stats.a2c_latency).rolling(smoothing_window,
                                                       min_periods=smoothing_window).mean()
    mask_ppo_latency = pd.Series(stats.mask_ppo_latency).rolling(smoothing_window,
                                                                 min_periods=smoothing_window).mean()
    deepsets_latency = pd.Series(stats.deepsets_latency).rolling(smoothing_window,
                                                                 min_periods=smoothing_window).mean()
    deepsets_dqn_latency = pd.Series(stats.deepsets_dqn_latency).rolling(smoothing_window,
                                                                         min_periods=smoothing_window).mean()

    a2c_cost = pd.Series(stats.a2c_cost).rolling(smoothing_window,
                                                 min_periods=smoothing_window).mean()
    mask_ppo_cost = pd.Series(stats.mask_ppo_cost).rolling(smoothing_window,
                                                           min_periods=smoothing_window).mean()
    deepsets_cost = pd.Series(stats.deepsets_cost).rolling(smoothing_window,
                                                           min_periods=smoothing_window).mean()
    deepsets_dqn_cost = pd.Series(stats.deepsets_dqn_cost).rolling(smoothing_window,
                                                                   min_periods=smoothing_window).mean()
    fig = plt.figure()
    plt.plot(a2c_rewards, color='#77AC30', label='A2C')
    plt.plot(mask_ppo_rewards, color='#D95319', label='Maskable PPO')
    plt.plot(deepsets_rewards, color='#3399FF', label='Deepsets PPO')
    plt.plot(deepsets_dqn_rewards, color='#EDB120', label='Deepsets DQN')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.legend()
    # plt.title("Episode Reward (Smoothed over window size {})".format(smoothing_window))
    plt.savefig(figName + '_reward.png', dpi=250, bbox_inches='tight')

    fig = plt.figure()
    plt.plot(a2c_ep_block_prob, color='#77AC30',label='A2C')
    plt.plot(mask_ppo_ep_block_prob, color='#D95319',label='Maskable PPO')
    plt.plot(deepsets_ep_block_prob, color='#3399FF',label='Deepsets PPO')
    plt.plot(deepsets_dqn_ep_block_prob, color='#EDB120', label='Deepsets DQN')
    plt.xlabel("Episode")
    plt.ylabel("Percentage of Rejected Requests")
    plt.xlim(0, 100)
    plt.ylim(0, 1)
    plt.legend()
    # plt.title("Episode Reward (Smoothed over window size {})".format(smoothing_window))
    plt.savefig(figName + '_block_probability.png', dpi=250, bbox_inches='tight')

    fig = plt.figure()
    plt.plot(a2c_latency, color='#77AC30',label='A2C')
    # plt.plot(ppo_sim_rewards, label='PPO (Simulation)')
    plt.plot(mask_ppo_latency, color='#D95319',label='Maskable PPO')
    # plt.plot(a2c_rewards, label='A2C (Cluster)')
    # plt.plot(ppo_rewards, label='PPO (Cluster)')
    plt.plot(deepsets_latency, color='#3399FF', label='Deepsets PPO')
    plt.plot(deepsets_dqn_latency, color='#EDB120',label='Deepsets DQN')
    plt.xlabel("Episode")
    plt.ylabel("Avg. Latency (in ms)")
    plt.xlim(0, 100)
    plt.ylim(0, 300)
    plt.legend()
    # plt.title("Episode Reward (Smoothed over window size {})".format(smoothing_window))
    plt.savefig(figName + '_latency.png', dpi=250, bbox_inches='tight')

    fig = plt.figure()
    plt.plot(a2c_cost, color='#77AC30',label='A2C')
    # plt.plot(ppo_sim_rewards, label='PPO (Simulation)')
    plt.plot(mask_ppo_cost, color='#D95319',label='Maskable PPO')
    # plt.plot(a2c_rewards, label='A2C (Cluster)')
    # plt.plot(ppo_rewards, label='PPO (Cluster)')
    plt.plot(deepsets_cost, color='#3399FF', label='Deepsets PPO')
    plt.plot(deepsets_dqn_cost, color='#EDB120', label='Deepsets DQN')
    plt.xlabel("Episode")
    plt.ylabel("Avg. Cost (in units)")
    plt.xlim(0, 100)
    plt.ylim(0, 18)
    plt.legend()
    # plt.title("Episode Reward (Smoothed over window size {})".format(smoothing_window))
    plt.savefig(figName + '_cost.png', dpi=250, bbox_inches='tight')

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


def print_statistics(df, alg_name):
    print("{} reward Mean: {}".format(alg_name, np.mean(df["reward"])))
    print("{} reward Std: {}".format(alg_name, np.std(df["reward"])))

    print("{} latency Mean: {}".format(alg_name, np.mean(df["avg_latency"])))
    print("{} latency Std: {}".format(alg_name, np.std(df["avg_latency"])))

    print("{} executionTime Mean: {}".format(alg_name, np.mean(df["executionTime"])))
    print("{} executionTime Std: {}".format(alg_name, np.std(df["executionTime"])))


if __name__ == "__main__":
    reward = 'risk'  # cost, risk or latency
    # num_algs = 4

    '''
    file_a2c = "results/karmada/" + reward + "/" \
                                             "a2c_env_karmada_num_clusters_4_reward_" + reward + "_totalSteps_200000_run_1" \
                                                                                                 "/karmada_gym_results.csv"
    file_mask_ppo = "results/karmada/" + reward + "/" \
                                                  "mask_ppo_env_karmada_num_clusters_4_reward_" + reward + "_totalSteps_200000_run_1" \
                                                                                                           "/karmada_gym_results.csv"
    file_deepsets_ppo = "results/karmada/" + reward + "/" \
                                                      "ppo_deepsets_env_karmada_num_clusters_4_reward_" + reward + "_totalSteps_200000_run_1" \
                                                                                                                   "/karmada_gym_results.csv"
    file_deepsets_dqn = "results/karmada/" + reward + "/" \
                                                      "dqn_deepsets_env_karmada_num_clusters_4_reward_" + reward + "_totalSteps_200000_run_1" \
                                                                                                               "/karmada_gym_results.csv"
    '''
    # testing
    file_a2c = "results/testing/" + reward + "/a2c/karmada_gym_results.csv"
    file_mask_ppo = "results/testing/" + reward + "/mask_ppo/karmada_gym_results.csv"
    file_deepsets_ppo = "results/testing/" + reward + "/ppo_deepsets/0_karmada_gym_results_num_clusters_4.csv"
    file_deepsets_dqn = "results/testing/" + reward + "/dqn_deepsets/0_karmada_gym_results_num_clusters_4.csv"

    df_a2c = pd.read_csv(file_a2c)
    df_mask_ppo = pd.read_csv(file_mask_ppo)
    df_deepsets_ppo = pd.read_csv(file_deepsets_ppo)
    df_deepsets_dqn = pd.read_csv(file_deepsets_dqn)

    # remove_empty_lines(df_a2c)
    # remove_empty_lines(df_mask_ppo)
    # remove_empty_lines(df_deepsets_ppo)
    # remove_empty_lines(df_deepsets_dqn)

    stats = stats(
        a2c_rewards=df_a2c['reward'],
        mask_ppo_rewards=df_mask_ppo['reward'],
        deepsets_rewards=df_deepsets_ppo['reward'],
        deepsets_dqn_rewards=df_deepsets_dqn['reward'],
        a2c_ep_block_prob=df_a2c['ep_block_prob'],
        mask_ppo_ep_block_prob=df_mask_ppo['ep_block_prob'],
        deepsets_ep_block_prob=df_deepsets_ppo['ep_block_prob'],
        deepsets_dqn_ep_block_prob=df_deepsets_dqn['ep_block_prob'],
        a2c_latency=df_a2c['avg_latency'],
        mask_ppo_latency=df_mask_ppo['avg_latency'],
        deepsets_latency=df_deepsets_ppo['avg_latency'],
        deepsets_dqn_latency=df_deepsets_dqn['avg_latency'],
        a2c_cost=df_a2c['avg_cost'],
        mask_ppo_cost=df_mask_ppo['avg_cost'],
        deepsets_cost=df_deepsets_ppo['avg_cost'],
        deepsets_dqn_cost=df_deepsets_dqn['avg_cost'],
    )

    plot_stats("karmada_testing_" + reward, stats, 5)

    print_statistics(df_a2c, "a2c")
    print_statistics(df_mask_ppo, "mask_ppo")
    print_statistics(df_deepsets_ppo, "deepsets_ppo")
    print_statistics(df_deepsets_dqn, "deepsets_dqn")
