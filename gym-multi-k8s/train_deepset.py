from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from envs.fog_env import FogOrchestrationEnv
from envs.ppo_deepset import PPO_DeepSets
from envs.karmada_scheduling_env import KarmadaSchedulingEnv

SEED = 2
MONITOR_PATH = f"./results/ppo_deepset_{SEED}_plot.monitor.csv"

if __name__ == "__main__":
    env = FogOrchestrationEnv(10, 100, 1)
    env.reset()
    _, _, _, info = env.step(0)
    info_keywords = tuple(info.keys())
    envs = SubprocVecEnv(
        [
            lambda: FogOrchestrationEnv(n_nodes=10, arrival_rate_r=100, call_duration_r=1, episode_length=100, seed=2)
            for i in range(8)
        ]
    )
    envs = VecMonitor(envs, MONITOR_PATH, info_keywords=info_keywords)

    '''
    # env = KarmadaSchedulingEnv(num_clusters=4, arrival_rate_r=100, call_duration_r=1, episode_length=100)
    # For faster training!
    # otherwise just comment the following lines
    #_, _, _, info = env.step(0)
    #info_keywords = tuple(info.keys())
    #envs = SubprocVecEnv([lambda: KarmadaSchedulingEnv(num_clusters=4, arrival_rate_r=100,
    #                                                  call_duration_r=1, episode_length=100,
    #                                                  reward_function='risk-aware') for i in range(8)])
    #envs = VecMonitor(env, info_keywords=info_keywords)
    '''
    # envs = VecNormalize(envs, norm_obs=True, norm_reward=False)
    agent = PPO_DeepSets(envs, num_steps=100, n_minibatches=8, ent_coef=0.001, tensorboard_log=None, seed=SEED)
    agent.learn(1500000)
    agent.save(f"./agents/ppo_deepset_{SEED}_plot.py")