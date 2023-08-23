import d3rlpy
import gym
import numpy as np
from d3rlpy.algos import DQNConfig, DiscreteSACConfig, LinearDecayEpsilonGreedy
from d3rlpy.dataset import ReplayBuffer
from gym.utils import seeding
from gym.vector.utils import spaces

from envs.deepSetAgent import CustomEncoderDQNFactory, CustomEncoderSACFactory
from envs.fog_env import FogOrchestrationEnv
from envs.karmada_scheduling_env import KarmadaSchedulingEnv

# prepare environment
# env = FogOrchestrationEnv(n_nodes=4, arrival_rate_r=100, call_duration_r=1, episode_length=100)
env = KarmadaSchedulingEnv(num_clusters=4, arrival_rate_r=100, call_duration_r=1, episode_length=100)

# eval_env = FogOrchestrationEnv(n_nodes=10, arrival_rate_r=100, call_duration_r=1, episode_length=100)

num_actions = env.action_space.n

# prepare algorithms
dqn = DQNConfig(encoder_factory=CustomEncoderDQNFactory(action_size=num_actions, feature_size=64)).create(device=None)

# sac = DiscreteSACConfig(
    # actor_encoder_factory=CustomEncoderSACFactory(action_size=num_actions, feature_size=64),
    # critic_encoder_factory=CustomEncoderSACFactory(action_size=num_actions, feature_size=64),
# ).create(device=None)

# standard prepare algorithm
# dqn = d3rlpy.algos.DQNConfig().create(device=None)

# prepare replay buffer
buffer = d3rlpy.dataset.create_fifo_replay_buffer(limit=1000000, env=env)

# start training
# dqn.fit_online(env, buffer, n_steps=1000000)
dqn.fit_online(env, buffer, n_steps=1000000) # eval_env=eval_env)