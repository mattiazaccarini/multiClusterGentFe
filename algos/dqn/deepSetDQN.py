import random
import time
import gym
import gym.spaces
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Union
from copy import deepcopy
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.buffers import ReplayBuffer

from drl_algos.common.algorithm import Algorithm
from drl_algos.dqn.agents import DQNDeepSetAgent 


def make_env(env_id, seed, rank):
    """
    Utility function for multiprocessed envs.
    
    :param env_id: (str) the environment ID
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)  # Replace this with your custom env if not using gym.make
        env.seed(seed + rank)  # Make sure each environment instance has a unique seed
        return env
    return _init

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


class DQN(Algorithm):
    def __init__(
        self,
        env: Union[SubprocVecEnv, DummyVecEnv],
        seed=1,
        torch_deterministic=True,
        learning_rate=2.5e-4,
        buffer_size=10000,
        gamma=0.99,
        tau=1.0,
        target_network_frequency=500,
        batch_size=128,
        start_e=1,
        end_e=0.05,
        exploration_fraction=0.5,
        learning_starts=10000,
        train_frequency=10,
        device: str = "cpu",
        tensorboard_log: str = "./run",
    ):
        super().__init__(env, learning_rate, seed, device, tensorboard_log)
        self.num_envs = env.num_envs
        self.seed = seed
        self.torch_deterministic = torch_deterministic
        self.env = env
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.tau = tau
        self.target_network_frequency = target_network_frequency
        self.batch_size = batch_size
        self.start_e = start_e
        self.end_e = end_e
        self.exploration_fraction = exploration_fraction
        self.learning_starts = learning_starts
        self.train_frequency = train_frequency


        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # TODO: modify it to multibinary
        assert isinstance(self.env.action_space, gym.spaces.Discrete), "only discrete action space is supported"
        
        # Initialize primary and target DQNDeepSetAgent
        self.q_network = DQNDeepSetAgent(self.env).to(self.device)
        self.target_network = deepcopy(self.q_network)  # Create a deep copy for the target network
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        # Initialize masks
        self.masks = torch.zeros((self.num_envs, self.env.action_space.n), dtype=torch.bool).to(self.device)

        # Initialize actions
        self.actions = torch.zeros((self.num_envs,) + self.env.action_space.shape).to(self.device)

        # Initialize replay buffer
        self.rb = ReplayBuffer(
            self.buffer_size,
            self.env.observation_space,
            self.env.action_space,
            self.device,
            self.num_envs,
            handle_timeout_termination=False,
        )

    def learn(self, total_timesteps: int = 500000):
        start_time = time.time()
        obs = self.env.reset()
        next_masks = torch.tensor(np.array(self.env.env_method("action_masks")), dtype=torch.bool).to(self.device)

        for global_step in range(total_timesteps):
            epsilon = linear_schedule(self.start_e, self.end_e, self.exploration_fraction * total_timesteps, global_step)
            self.masks = next_masks

            if random.random() < epsilon:
                actions = []
                for env_mask in self.masks:
                    valid_actions = np.where(env_mask)[0]
                    action = np.random.choice(valid_actions)
                    actions.append(action)
                self.actions = torch.tensor(actions).to(self.device)
            else:
                q_values = self.q_network(torch.Tensor(obs).to(self.device))
                
                # Masking Q-values of invalid actions
                HUGE_NEGATIVE = -1e8
                action_masks_tensor = torch.tensor(self.masks, dtype=torch.bool).to(self.device)
                q_values = torch.where(action_masks_tensor, q_values, HUGE_NEGATIVE)

                self.actions = torch.argmax(q_values, dim=1)
            # Execute the game and log data
            next_obs, rewards, terminated, infos = self.env.step(self.actions.cpu().numpy())
            next_masks = torch.tensor(np.array(self.env.env_method("action_masks")), dtype=torch.bool).to(self.device)
            

            # Record rewards for plotting purposes
            for item in infos:
                    if "episode" in item.keys():
                        print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                        break

            # Save data to replay buffer; handle `final_observation`
            real_next_obs = next_obs.copy()
            
            #Since we have an environment with a fixed episode length, we don't much need to handle the truncated cases
            self.rb.add(obs, real_next_obs, self.actions.cpu().numpy().reshape(-1, 1), rewards, terminated, infos)

            #return next_obs
            obs = next_obs

            # ALGO LOGIC: training
            if global_step > self.learning_starts:
                if global_step % self.train_frequency == 0:
                    data = self.rb.sample(self.batch_size)
                    with torch.no_grad():
                        target_max, _ = self.target_network(data.next_observations).max(dim=1)
                        td_target = data.rewards.flatten() + self.gamma * target_max * (1 - data.dones.flatten())
                    old_val = self.q_network(data.observations).gather(1, data.actions).squeeze()
                    loss = F.mse_loss(td_target, old_val)

                    '''
                    if global_step % 100 == 0:
                        writer.add_scalar("losses/td_loss", loss, global_step)
                        writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                        print("SPS:", int(global_step / (time.time() - start_time)))
                        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                    '''
                    
                    # optimize the model
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Update target network
                    if global_step % self.target_network_frequency == 0:
                        for target_param, q_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                            target_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_param.data)

            #print("FPS:", int(global_step / (time.time() - start_time)))
            
    def predict(self, obs: npt.NDArray, masks: Optional[npt.NDArray] = None) -> npt.NDArray:
        with torch.no_grad():
            action = self.q_network.get_action(
                torch.as_tensor(obs, dtype=torch.float32), torch.as_tensor(masks, dtype=torch.bool), deterministic=True
            ).numpy()
        return action
    
    
    def save(self, path: str) -> None:
        torch.save(self.q_network.state_dict(), path)

    def load(self, path: str) -> None:
        self.q_network.load_state_dict(torch.load(path))
