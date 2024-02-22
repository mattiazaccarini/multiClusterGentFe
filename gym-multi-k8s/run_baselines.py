import logging

import numpy as np
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from envs.karmada_scheduling_env import KarmadaSchedulingEnv
from envs.utils import resource_greedy_policy, latency_greedy_policy, cost_greedy_policy

MONITOR_PATH = "./results/lb_qos_greedy.monitor.csv"

# Logging
logging.basicConfig(filename='run_baselines.log', filemode='w', level=logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

RESOURCE_GREEDY = 'resource'
LATENCY_GREEDY = 'latency'
COST_GREEDY = 'cost'

if __name__ == "__main__":
    policy = RESOURCE_GREEDY

    env = KarmadaSchedulingEnv(num_clusters=4, arrival_rate_r=100, call_duration_r=1,
                               episode_length=100,
                               reward_function='cost')
    env.reset()
    _, _, _, info = env.step(0)
    info_keywords = tuple(info.keys())
    env = KarmadaSchedulingEnv(num_clusters=4, arrival_rate_r=100, call_duration_r=1,
                               episode_length=100,
                               reward_function='cost')
    # env = Monitor(env, filename=MONITOR_PATH, info_keywords=info_keywords)

    returns = []
    for _ in tqdm(range(2000)):
        obs = env.reset()
        action_mask = env.action_masks()
        return_ = 0.0
        done = False
        while not done:
            if policy == RESOURCE_GREEDY:
                action = resource_greedy_policy(env, obs, action_mask, env.latency, env.deployment_request.latency_threshold)
            elif policy == LATENCY_GREEDY:
                action = latency_greedy_policy(action_mask, env.latency, env.deployment_request.latency_threshold)
            elif policy == COST_GREEDY:
                action = cost_greedy_policy(env, action_mask)
            else:
                print("unrecognized policy!")

            obs, reward, done, info = env.step(action)
            action_mask = env.action_masks()
            return_ += reward
        returns.append(return_)

    print(f"{np.mean(returns)} +/- {1.96 * np.std(returns) / np.sqrt(len(returns))}")