from typing import Optional
import d3rlpy
import gym
import numpy as np
import torch
import torch.nn as nn
import pdb
from torch.distributions import Categorical


def layer_init(layer: nn.Linear, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class MLPAgent(nn.Module):
    def __init__(self, envs: gym.vector.VectorEnv):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
        return self.critic(x)

    def get_action(self, x: torch.Tensor, deterministic=True) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
        logits = self.actor(x)
        dist = Categorical(logits=logits)
        if deterministic:
            return dist.mode()
        return dist.sample()

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.flatten(x, start_dim=1)
        logits = self.actor(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(x)


class EquivariantLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.Gamma = nn.Linear(in_channels, out_channels, bias=False)
        self.Lambda = nn.Linear(in_channels, out_channels, bias=False)
    
    def forward(self, x: torch.Tensor):
        #pdb.set_trace()
        gamma_x = self.Gamma(x)
        xm, _ = torch.max(gamma_x, dim=1, keepdim=False)
        return self.Lambda(x) - self.Gamma(xm.unsqueeze(1).expand_as(x))
    

class EquivariantDeepSet(nn.Module):
    def __init__(self, in_channels, num_actions, hidden_channels: int = 64):
        #pdb.set_trace()
        super().__init__()
        self.final_layer = nn.Linear(num_actions, hidden_channels)

        self.net = nn.Sequential(
            EquivariantLayer(in_channels, hidden_channels),
            nn.ReLU(),
            EquivariantLayer(hidden_channels, hidden_channels),
            nn.ELU(),
            EquivariantLayer(hidden_channels, num_actions),
        )
    
    def forward(self, x: torch.Tensor):
        #return torch.squeeze(self.net(x), dim=-1)
        x = self.net(x)
        return self.final_layer(x)


class InvariantDeepSet(nn.Module):
    def __init__(self, in_channels, hidden_channels: int = 64):
        super().__init__()
        self.psi = nn.Sequential(
            EquivariantLayer(in_channels, hidden_channels),
            nn.ELU(),
            EquivariantLayer(hidden_channels, hidden_channels),
            nn.ELU(),
            EquivariantLayer(hidden_channels, hidden_channels)
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ELU(),
            nn.Linear(hidden_channels, 1)
        )
    
    def forward(self, x: torch.Tensor):
        x = torch.mean(self.psi(x), dim=1)
        return torch.squeeze(self.rho(self.psi(x)), dim=-1)
    

class DeepSetAgent(nn.Module):
    def __init__(self, observation_shape, num_actions, feature_size):
        super().__init__()
        in_channels = observation_shape[0]
        self.feature_size = feature_size

        '''
        # Actor outputs pi(a|s)
        self.actor = EquivariantDeepSet(in_channels, feature_size)
        # Critic outputs V(s)
        self.critic = InvariantDeepSet(in_channels)
        '''

        # Only one network for DQN which outputs Q-values
        self.q_network = EquivariantDeepSet(in_channels, num_actions, feature_size)


    def forward(self, x: torch.Tensor):
        return self.q_network(x)

    def get_value(self, x: torch.Tensor):
        return self.critic(x)

    def get_action(self, x: torch.Tensor, masks: Optional[torch.Tensor] = None, deterministic: bool = True):
        logits = self.actor(x)
        if masks is not None:
            HUGE_NEG = torch.tensor(-1e8, dtype=logits.dtype)
            logits = torch.where(masks, logits, HUGE_NEG)
        # discrete probability distribution over a set of actions. The logits provide the unnormalized log probabilities for each action.
        dist = Categorical(logits=logits)
        # if deterministic is True, return the mode of the Categorical distribution (highest probability, selecting the action with the highest logit value)
        if deterministic:
            return dist.mode
        # if deterministic is False, return a random sample from the Categorical distribution.
        return dist.sample()

    def get_action_and_value(self, x:torch.Tensor, action: Optional[torch.Tensor], masks: Optional[torch.Tensor] = None):
        logits = self.actor(x)
        if masks is not None:
            HUGE_NEG = torch.tensor(-1e8, dtype=logits.dtype)
            logits = torch.where(masks, logits, HUGE_NEG)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(x)
    
    def get_feature_size(self):
        return self.feature_size


class ActionConditionedDeepSet(nn.Module):
    def __init__(self, observation_shape, action_size, feature_size):
        super().__init__()
        self.feature_size = feature_size
        self.action_size = action_size
        
        # Equivariant deep set with updated input channels to account for action
        self.equivariant = EquivariantDeepSet(observation_shape[0] + action_size, action_size, feature_size)
        
        # Invariant deep set
        self.invariant = InvariantDeepSet(feature_size)
        
        # Additional layers can be added if necessary, for example:
        self.fc = nn.Linear(feature_size, feature_size)
        self.q_value_layer = nn.Linear(feature_size, 1)  # for Q-value output
        
    def forward(self, x, action):
        # Concatenate action to each observation in the set
        h = torch.cat([x, action.unsqueeze(1).repeat(1, x.shape[1], 1)], dim=-1)
        
        # Pass through equivariant and invariant deep sets
        h = self.equivariant(h)
        h = self.invariant(h)
        
        # Additional processing
        h = torch.relu(self.fc(h))
        q_value = self.q_value_layer(h)
        
        return q_value

    def get_feature_size(self):
        return self.feature_size
    

class CustomEncoderDQNFactory(d3rlpy.models.encoders.EncoderFactory):
    TYPE = "custom"  # this is necessary

    def __init__(self, feature_size, action_size):
        self.feature_size = feature_size
        self.action_size = action_size
        
    def create(self, observation_shape):
        return DeepSetAgent(observation_shape, self.action_size, self.feature_size)

    def get_params(self, deep=False):
        return {"feature_size": self.feature_size}
    
    def get_type(self):
        return "DeepSetAgent"


class CustomEncoderSACFactory(d3rlpy.models.encoders.EncoderFactory):
    TYPE = "custom"

    def __init__(self, action_size, feature_size):
        self.feature_size = feature_size
        self.action_size = action_size

    def create(self, observation_shape):
        return DeepSetAgent(observation_shape, self.action_size, self.feature_size)

    def create_with_action(self, observation_shape):
        return ActionConditionedDeepSet(observation_shape, self.action_size, self.feature_size)

    def get_params(self, deep=False):
        return {"feature_size": self.feature_size}
    
    def get_type(self):
        return "ActionConditionedDeepSet"
 