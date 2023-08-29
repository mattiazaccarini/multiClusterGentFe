import d3rlpy
from d3rlpy.algos import DQNConfig, DQN, LinearDecayEpsilonGreedy
from d3rlpy.dataset import ReplayBuffer
from d3rlpy.datasets import get_cartpole

from envs.karmada_scheduling_env import KarmadaSchedulingEnv
from envs.deepSetAgent import CustomEncoderDQNFactory

SEED = 2
MONITOR_PATH = f"./results/karmada/ppo_mask/ppo_mask_sb3_{SEED}_plot.monitor.csv"


# prepare environment
# env = FogOrchestrationEnv(n_nodes=4, arrival_rate_r=100, call_duration_r=1, episode_length=100)

env = KarmadaSchedulingEnv(num_clusters=4, arrival_rate_r=100, call_duration_r=1, episode_length=100)
eval_env = KarmadaSchedulingEnv(num_clusters=4, arrival_rate_r=100, call_duration_r=1, episode_length=100)
# _, _, _, info = env.step(0)
# info_keywords = tuple(info.keys())
# envs = SubprocVecEnv([lambda: KarmadaSchedulingEnv(num_clusters=4, arrival_rate_r=100, call_duration_r=1, episode_length=100) for i in range(4)])
# envs = VecMonitor(envs, MONITOR_PATH, info_keywords=info_keywords)

num_actions = env.action_space.n
print(num_actions)

# prepare algorithm
alg = d3rlpy.algos.BCQConfig().create(device=None)

# prepare replay buffer
buffer = d3rlpy.dataset.create_fifo_replay_buffer(limit=1000000, env=env)

# train online
alg.fit_online(env, buffer, n_steps=1000000)

# ready to control
# actions = sac.predict(x)

# prepare algorithms
# dqn = DQNConfig(encoder_factory=CustomEncoderDQNFactory(action_size=num_actions, feature_size=8)).create(device=None)


#sac = DiscreteSACConfig(
#    actor_encoder_factory=CustomEncoderSACFactory(action_size=num_actions, feature_size=64),
#    critic_encoder_factory=CustomEncoderSACFactory(action_size=num_actions, feature_size=64),
#).create(device=None)

# standard prepare algorithm
# dqn = d3rlpy.algos.DQNConfig().create(device=None)


# start training
# sac.fit_online(env, buffer, n_steps=1000000, eval_env=eval_env)
# dqn.fit_online(env, buffer, n_steps=1000000)
