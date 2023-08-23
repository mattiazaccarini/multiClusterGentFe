#https://d3rlpy.readthedocs.io/en/latest/tutorials/getting_started.html

import d3rlpy 
import numpy as np
from d3rlpy.datasets import get_cartpole
from d3rlpy.algos import DQN, DQNConfig

from envs.deepSetAgent import CustomEncoderDQNFactory
from d3rlpy.metrics import TDErrorEvaluator
from d3rlpy.metrics import EnvironmentEvaluator
from envs.fog_env import FogOrchestrationEnv

dataset, env = get_cartpole()
num_actions = env.action_space.n

dqn = DQNConfig(encoder_factory=CustomEncoderDQNFactory(action_size=num_actions, feature_size=64)).create(device=None)

#initialize neural networks with the given observation shape and action size.
#not necessary when you directly call fit or fit_online method.
dqn.build_with_dataset(dataset)

td_error_evaluator = TDErrorEvaluator(episodes=dataset.episodes)
environment_evaluator = EnvironmentEvaluator(env) # set environment in scorer function
rewards = environment_evaluator(dqn, dataset=None) # evaluate algorithm on the environment


dqn.fit(dataset,
        n_steps=10000,
        evaluators={
            'environment': environment_evaluator,
            # 'td_error': td_error_evaluator,
        })



'''
observation = env.reset()

# return actions based on the greedy-policy
action = dqn.predict(np.expand_dims(observation[0], axis=0))

# estimate action-values
value = dqn.predict_value(np.expand_dims(observation[0], axis=0), action)

# save full parameters and configurations in a single file.
dqn.save('dqn.d3')
# load full parameters and build algorithm
dqn2 = d3rlpy.load_learnable("dqn.d3")
# save full parameters only
dqn.save_model('dqn.pt')
# load full parameters with manual setup
dqn3 = DQNConfig().create(device=None)
dqn3.build_with_dataset(dataset)
dqn3.load_model('dqn.pt')
# save the greedy-policy as TorchScript
dqn.save_policy('policy.pt')
# save the greedy-policy as ONNX
dqn.save_policy('policy.onnx')

#if you want to use GPU, set device to "cuda:0"
#dqn = DQNConfig().create(device=None)
#sac = DiscreteSACConfig().create(device=None)
'''