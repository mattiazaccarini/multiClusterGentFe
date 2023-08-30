import d3rlpy 
import numpy as np

from d3rlpy.datasets import get_cartpole
from d3rlpy.algos import DiscreteSACConfig

from deepSetAgent import CustomEncoderSACFactory
from d3rlpy.metrics import TDErrorEvaluator
from d3rlpy.metrics import EnvironmentEvaluator

dataset, env = get_cartpole()

num_actions = env.action_space.n

sac = DiscreteSACConfig(
    actor_encoder_factory=CustomEncoderSACFactory(action_size=num_actions, feature_size=64),
    critic_encoder_factory=CustomEncoderSACFactory(action_size=num_actions, feature_size=64),
).create(device=None)

sac.build_with_dataset(dataset)

td_error_evaluator = TDErrorEvaluator(episodes=dataset.episodes)
environment_evaluator = EnvironmentEvaluator(env) # set environment in scorer function
rewards = environment_evaluator(sac, dataset=None) # evaluate algorithm on the environment

sac.fit(dataset, 
        n_steps=10000,
        evaluators={
            'environment': environment_evaluator,
            'td_error': td_error_evaluator,
        })
