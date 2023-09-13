import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from tqdm import tqdm
from envs.karmada_scheduling_env import KarmadaSchedulingEnv
from envs.dqn_deepset import DQN_DeepSets
from envs.ppo_deepset import PPO_DeepSets

SEED = 2
env_kwargs = {"n_nodes": 10, "arrival_rate_r": 100, "call_duration_r": 1, "episode_length": 100}
MONITOR_PATH = f"./results/test/ppo_deepset_{SEED}_n{env_kwargs['n_nodes']}_lam{env_kwargs['arrival_rate_r']}_mu{env_kwargs['call_duration_r']}.monitor.csv"

if __name__ == "__main__":
    num_clusters = 4
    env = KarmadaSchedulingEnv(num_clusters=num_clusters, arrival_rate_r=100, call_duration_r=1,
                                   episode_length=100,
                                   reward_function='cost')
    env.reset()
    _, _, _, info = env.step(0)
    info_keywords = tuple(info.keys())

    envs = DummyVecEnv([lambda: KarmadaSchedulingEnv(num_clusters=num_clusters, arrival_rate_r=100, call_duration_r=1,
                                   episode_length=100,
                                   reward_function='cost')])
    envs = VecMonitor(envs, MONITOR_PATH, info_keywords=info_keywords)

    agent = PPO_DeepSets(envs, seed=SEED, tensorboard_log=None)
    # agent = DQN_DeepSets(envs, seed=SEED, tensorboard_log=None)
    agent.load(f"./results/karmada/cost/ppo_deepsets_env_karmada_num_clusters_4_reward_cost_totalSteps_200000_run_1/"
               f"ppo_deepsets_env_karmada_num_clusters_4_reward_cost_totalSteps_200000")

    #agent.load(f"./agents/ppo_deepset_{SEED}")
    for _ in tqdm(range(100)):
        obs = envs.reset()
        action_mask = np.array(envs.env_method("action_masks"))
        done = False
        while not done:
            action = agent.predict(obs, action_mask)
            obs, reward, dones, info = envs.step(action)
            action_mask = np.array(envs.env_method("action_masks"))
            done = dones[0]