import csv
import math
from datetime import datetime
import heapq
import time
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from envs.utils import DeploymentRequest, Cluster, get_c2e_deployment_list, save_to_csv
import logging

# Number of steps per episode
DEFAULT_NUM_EPISODE_STEPS = 25

# MAX Number of Replicas per deployment request
MAX_REPLICAS = 8

# Actions - to modify accordingly
ACTIONS = ["All-", "Divide", "Reject"]

# Reward objectives - to modify after accordingly
LATENCY = 'latency'
COST = 'cost'


class KarmadaSchedulingEnv(gym.Env):
    """ Karmada Scheduling env in Kubernetes - an OpenAI gym environment"""
    metadata = {'render.modes': ['human', 'ansi', 'array']}

    def __init__(self, num_clusters=4, arrival_rate_r=100, call_duration_r=1,
                 episode_length=DEFAULT_NUM_EPISODE_STEPS, goal_reward=COST):
        # Define action and observation space

        super(KarmadaSchedulingEnv, self).__init__()
        self.name = "karmada_gym"
        self.__version__ = "0.0.1"
        self.goal_reward = goal_reward

        self.num_clusters = num_clusters
        self.arrival_rate_r = arrival_rate_r
        self.call_duration_r = call_duration_r
        self.episode_length = episode_length
        self.running_requests: list[DeploymentRequest] = []

        self.seed = 42
        self.np_random, seed = seeding.np_random(self.seed)

        logging.info("[Init] Env: {} | Version {} |".format(self.name, self.__version__))

        # Defined as a matrix having as rows the nodes and columns their associated metrics
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(num_clusters + 1, 8),
                                            dtype=np.float32)

        # Action Space
        # deploy the service on cluster 1,2,..., n + split replicas + reject it
        self.num_actions = num_clusters + 2

        # Discrete version
        self.action_space = spaces.Discrete(self.num_actions)

        # Action and Observation Space
        logging.info("[Init] Action Space: " + str(self.action_space))
        logging.info("[Init] Observation Space: " + str(self.observation_space))

        # Setting the experiment based on Cloud2Edge (C2E) deployments
        self.deploymentList = get_c2e_deployment_list()
        self.deployment_request = None

        # Resource capacity
        # TODO: Perhaps add Storage as well later
        self.cpu_capacity = self.np_random.integers(low=2.0, high=6.0, size=num_clusters)
        self.memory_capacity = self.np_random.integers(low=2.0, high=6.0, size=num_clusters)

        # Keeps track of allocated resources
        self.allocated_cpu = self.np_random.uniform(low=0.0, high=0.2, size=num_clusters)
        self.allocated_memory = self.np_random.uniform(low=0.0, high=0.2, size=num_clusters)

        # Keeps track of Free resources for deployment requests
        self.free_cpu = np.zeros(num_clusters)
        self.free_memory = np.zeros(num_clusters)
        for n in range(num_clusters):
            self.free_cpu[n] = self.cpu_capacity[n] - self.allocated_cpu[n]
            self.free_memory[n] = self.memory_capacity[n] - self.allocated_memory[n]

        # Variables for divide strategy
        self.split_number_replicas = np.zeros(num_clusters)
        self.calculated_split_number_replicas = np.zeros(num_clusters)

        # TODO: print correctly
        logging.info("[Init] Resources:")
        logging.info("[Init] CPU Capacity: {}".format(self.cpu_capacity))
        logging.info("[Init] CPU allocated: {}".format(self.allocated_cpu))
        logging.info("[Init] CPU free: {}".format(self.free_cpu))

        logging.info("[Init] MEM Capacity: {}".format(self.memory_capacity))
        logging.info("[Init] MEM allocated: {}".format(self.allocated_memory))
        logging.info("[Init] MEM free: {}".format(self.free_memory))

        # Current Step
        self.current_step = 0
        self.current_time = 0
        self.penalty = False

        self.accepted_requests = 0
        self.offered_requests = 0
        self.ep_accepted_requests = 0
        self.next_request()

        # Info & episode over
        self.total_reward = 0
        self.episode_over = False
        self.info = {}

        self.time_start = 0
        self.execution_time = 0
        self.episode_count = 0
        self.file_results = self.name + "_results.csv"
        self.obs_csv = self.name + "_obs.csv"

    def step(self, action):
        if self.current_step == 1:
            self.time_start = time.time()

        # Execute one time step within the environment
        self.offered_requests += 1
        self.take_action(action)

        # Update observation before reward calculation
        ob = self.get_state()

        # Calculate Reward
        reward = self.get_reward()
        self.total_reward += reward

        # Find correct action move for logging purposes
        move = ""
        if action < self.num_clusters:
            move = ACTIONS[0] + "cluster-" + str(action + 1)
        elif action == self.num_clusters:
            move = ACTIONS[1]
        else:
            move = ACTIONS[2]

        # Logging Step and Total Reward
        logging.info('[Step {}] | Action: {} | Reward: {} | Total Reward: {}'.format(
            self.current_step, move, reward, self.total_reward))

        # Get next request
        self.next_request()

        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # self.save_obs_to_csv(self.obs_csv, np.array(ob), date)

        self.info = {
            "reward": reward,
            "action": action,
            "block_prob": 1 - (self.accepted_requests / self.offered_requests),
            "ep_block_prob": 1 - (self.ep_accepted_requests / self.current_step),
        }

        if self.current_step == self.episode_length:
            self.episode_count += 1
            self.episode_over = True
            self.execution_time = time.time() - self.time_start
            logging.info("[Step] Episode finished, saving results to csv...")
            save_to_csv(self.file_results, self.episode_count,
                        self.total_reward, self.execution_time)

        # return ob, reward, self.episode_over, self.info
        return np.array(ob), reward, self.episode_over, self.info

    # TODO: update reward function based on Multi-objective function
    def get_reward(self):
        """ Calculate Rewards """
        if self.penalty:
            reward = -1
        else:
            reward = 1

        return reward

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.current_step = 0
        self.episode_over = False
        self.total_reward = 0
        self.ep_accepted_requests = 0
        self.penalty = False

        # Reset Deployment Data
        self.deploymentList = get_c2e_deployment_list()

        # Reset Resources
        # TODO: Perhaps add Storage as well later
        self.cpu_capacity = self.np_random.integers(low=2.0, high=6.0, size=self.num_clusters)
        self.memory_capacity = self.np_random.integers(low=2.0, high=6.0, size=self.num_clusters)

        self.allocated_cpu = self.np_random.uniform(low=0.0, high=0.2, size=self.num_clusters)
        self.allocated_memory = self.np_random.uniform(low=0.0, high=0.2, size=self.num_clusters)

        self.free_cpu = np.zeros(self.num_clusters)
        self.free_memory = np.zeros(self.num_clusters)
        for n in range(self.num_clusters):
            self.free_cpu[n] = self.cpu_capacity[n] - self.allocated_cpu[n]
            self.free_memory[n] = self.memory_capacity[n] - self.allocated_memory[n]

        # Variables for divide strategy
        self.split_number_replicas = np.zeros(self.num_clusters)
        self.calculated_split_number_replicas = np.zeros(self.num_clusters)

        # return obs
        return np.array(self.get_state())

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        return

    def take_action(self, action):
        self.current_step += 1

        # Stop if MAX_STEPS
        if self.current_step == self.episode_length:
            # logging.info('[Take Action] MAX STEPS achieved, ending ...')
            self.episode_over = True

        # Possible Actions: Place all replicas together or split them.
        # Known as NP-hard problem (Bin pack with fragmentation)
        # Any ideas for heuristic? We can later compare with an ILP/MILP model...
        # Check first if "Place all" Action can be performed
        if action < self.num_clusters:
            if self.check_if_cluster_is_full_after_full_deployment(action):
                self.penalty = True
                logging.info('[Take Action] Block the selected action since cluster will be full!')
                # Do not raise error since algorithm might not support action mask
                # raise ValueError("Action mask is not working properly. Full nodes should be always masked.")
            else:
                # accept request
                self.accepted_requests += 1
                self.ep_accepted_requests += 1
                self.deployment_request.deployed_cluster = action
                # Update allocated amounts
                self.allocated_cpu[action] += self.deployment_request.cpu_request * self.deployment_request.num_replicas
                self.allocated_memory[action] += self.deployment_request.memory_request * self.deployment_request.num_replicas
                # Update free resources
                self.free_cpu[action] = self.cpu_capacity[action] - self.allocated_cpu[action]
                self.free_memory[action] = self.memory_capacity[action] - self.allocated_memory[action]
                self.enqueue_request(self.deployment_request)

        # Divide Strategy selected
        # TODO: define divide strategy based on heuristic
        elif action == self.num_clusters:
            if self.deployment_request.num_replicas == 1:
                logging.info('[Take Action] Block Divide strategy since only one replica... ')
                self.penalty = True
            else:
                # logging.info('[Take Action] Divide strategy chosen... ')
                self.penalty = False

                div = self.distribute_replicas_in_clusters(self.deployment_request.num_replicas,
                                                           self.deployment_request.cpu_request,
                                                           self.deployment_request.memory_request, self.num_clusters,
                                                           self.free_cpu, self.free_memory)
                # logging.info('[Divide] division: {}'.format(div))

                if self.check_if_clusters_are_full_after_split_deployment(div):
                    self.penalty = True
                    logging.info('[Take Action] Block the divide strategy since cluster will be full!')
                else:
                    # accept request
                    self.penalty = False
                    self.accepted_requests += 1
                    self.ep_accepted_requests += 1
                    self.deployment_request.split_clusters = div
                    self.deployment_request.is_deployment_split = True

                    # logging.info("[Divide] Before")
                    # logging.info("[Divide] CPU allocated: {}".format(self.allocated_cpu))
                    # logging.info("[Divide] CPU free: {}".format(self.free_cpu))
                    # logging.info("[Divide] MEM allocated: {}".format(self.allocated_memory))
                    # logging.info("[Divide] MEM free: {}".format(self.free_memory))

                    for d in range(len(div)):
                        # Update allocated amounts
                        self.allocated_cpu[d] += self.deployment_request.cpu_request * div[d]
                        self.allocated_memory[d] += self.deployment_request.memory_request * div[d]

                        # Update free resources
                        self.free_cpu[d] = self.cpu_capacity[d] - self.allocated_cpu[d]
                        self.free_memory[d] = self.memory_capacity[d] - self.allocated_memory[d]

                    # logging.info("[Divide] After")
                    # logging.info("[Divide] CPU allocated: {}".format(self.allocated_cpu))
                    # logging.info("[Divide] CPU free: {}".format(self.free_cpu))

                    # logging.info("[Divide] MEM allocated: {}".format(self.allocated_memory))
                    # logging.info("[Divide] MEM free: {}".format(self.free_memory))
                    self.enqueue_request(self.deployment_request)

        # Reject the request: give the agent a penalty, especially if the request could have been accepted
        elif action == self.num_clusters + 1:
            self.penalty = True
        else:
            logging.info('[Take Action] Unrecognized Action: ' + str(action))

    # Current Strategy: Distribute evenly. Improved algorithm can be considered
    def distribute_replicas_in_clusters(self, num_replicas, cpu_req, mem_req, num_clusters, free_cpu, free_mem):
        # logging.info('[Divide] Num. replicas to distribute: {}'.format(num_replicas))
        distribution = [0] * num_clusters

        # min and max replicas
        min_replicas = 1
        max_replicas = num_replicas

        # Distribute the replicas across clusters
        for n in range(num_clusters):
            self.split_number_replicas[n] = min(free_cpu[n] / cpu_req,
                                                free_mem[n] / mem_req)

        # logging.info('[Take Action] Split factors: {}'.format(self.split_number_replicas))
        min_factor = int(math.ceil(min(self.split_number_replicas)))
        # logging.info('[Take Action] Min factor: {}'.format(min_factor))

        if min_factor >= max_replicas:
            min_factor = max_replicas - 1  # To really distribute at the end

        for n in range(num_clusters):
            if num_replicas > 0 and (cpu_req < free_cpu[n] and mem_req < free_mem[n]):
                if min_factor < num_replicas:
                    distribution[n] += min_factor
                    num_replicas -= min_factor
                else:
                    distribution[n] += min_replicas
                    num_replicas -= min_replicas
        return distribution

    def get_state(self):
        # Get Observation state
        cluster = np.full((1, 4), -1)
        observation = np.stack([self.allocated_cpu, self.cpu_capacity, self.allocated_memory, self.memory_capacity],
                               axis=1)
        # Condition the elements in the set with the current node request
        request_demands = np.tile(
            np.array(
                [self.deployment_request.num_replicas,
                 self.deployment_request.cpu_request,
                 self.deployment_request.memory_request,
                 self.dt]
            ),
            (self.num_clusters + 1, 1),
        )
        # TODO: concatenation fails here if + 2 is used in the obs space. It always needs to match!
        observation = np.concatenate([observation, cluster], axis=0)
        observation = np.concatenate([observation, request_demands], axis=1)
        # logging.info('[Get Obs State]: obs: {}'.format(observation))
        return observation


    def save_obs_to_csv(self, obs_file, obs, date):
        file = open(obs_file, 'a+', newline='')  # append
        # file = open(file_name, 'w', newline='') # new
        fields = []
        cluster_obs = {}
        with file:
            fields.append('date')
            for n in range(self.num_clusters):
                fields.append("cluster_" + str(n + 1) + '_allocated_cpu')
                fields.append("cluster_" + str(n + 1) + '_cpu_capacity')
                fields.append("cluster_" + str(n + 1) + '_allocated_memory')
                fields.append("cluster_" + str(n + 1) + '_memory_capacity')
                fields.append("cluster_" + str(n + 1) + '_num_replicas')
                fields.append("cluster_" + str(n + 1) + '_cpu_request')
                fields.append("cluster_" + str(n + 1) + '_memory_request')
                fields.append("cluster_" + str(n + 1) + '_dt')

            # logging.info("[Save Obs] fields: {}".format(fields))

            writer = csv.DictWriter(file, fieldnames=fields)
            # writer.writeheader() # write header

            cluster_obs = {}
            cluster_obs.update({fields[0]: date})

            for n in range(self.num_clusters):
                i = self.get_iteration_number(n)
                cluster_obs.update({fields[i+1]: obs[n][0]})
                cluster_obs.update({fields[i+2]: obs[n][1]})
                cluster_obs.update({fields[i+3]: obs[n][2]})
                cluster_obs.update({fields[i+4]: obs[n][3]})
                cluster_obs.update({fields[i+5]: obs[n][4]})
                cluster_obs.update({fields[i+6]: obs[n][5]})
                cluster_obs.update({fields[i+7]: obs[n][6]})
                cluster_obs.update({fields[i+8]: obs[n][7]})
            writer.writerow(cluster_obs)
        return

    def get_iteration_number(self, n):
        num_fields_per_cluster = 8
        return num_fields_per_cluster * n

    def enqueue_request(self, request: DeploymentRequest) -> None:
        heapq.heappush(self.running_requests, (request.departure_time, request))

    def action_masks(self):
        valid_actions = np.ones(self.num_clusters + 2, dtype=bool)
        for i in range(self.num_clusters):
            if self.check_if_cluster_is_full_after_full_deployment(i):
                valid_actions[i] = False
            else:
                valid_actions[i] = True

        # 2 additional actions: Divide and Reject
        valid_actions[self.num_clusters] = True
        valid_actions[self.num_clusters + 1] = True
        logging.info('[Action Mask]: Valid actions {} |'.format(valid_actions))
        return valid_actions

    def check_if_cluster_is_full_after_full_deployment(self, action):
        total_cpu = self.deployment_request.num_replicas * self.deployment_request.cpu_request
        total_memory = self.deployment_request.num_replicas * self.deployment_request.memory_request

        if (self.allocated_cpu[action] + total_cpu > 0.95 * self.cpu_capacity[action]
                or self.allocated_memory[action] + total_memory > 0.95 * self.memory_capacity[action]):
            logging.info('[Check]: Cluster {} is full...'.format(action+1))
            return True

        return False

    def check_if_clusters_are_full_after_split_deployment(self, div):
        for d in range(len(div)):
            total_cpu = self.deployment_request.cpu_request * div[d]
            total_memory = self.deployment_request.memory_request * div[d]

            if (self.allocated_cpu[d] + total_cpu > 0.95 * self.cpu_capacity[d]
                or self.allocated_memory[d] + total_memory > 0.95 * self.memory_capacity[d]):
                logging.info('[Check]: Cluster {} is full...'.format(d))
                return True

        return False

    def dequeue_request(self):
        _, deployment_request = heapq.heappop(self.running_requests)
        # logging.info("[Dequeue] Request {}...".format(deployment_request))
        # logging.info("[Dequeue] Request will be terminated...")
        # logging.info("[Dequeue] Before: ")
        # logging.info("[Dequeue] CPU allocated: {}".format(self.allocated_cpu))
        # logging.info("[Dequeue] CPU free: {}".format(self.free_cpu))
        # logging.info("[Dequeue] MEM allocated: {}".format(self.allocated_memory))
        # logging.info("[Dequeue] MEM free: {}".format(self.free_memory))

        if deployment_request.is_deployment_split:
            # logging.info("[Dequeue] Deployment is split...")
            for d in range(self.num_clusters):
                total_cpu = self.deployment_request.cpu_request * self.deployment_request.split_clusters[d]
                total_memory = self.deployment_request.memory_request * self.deployment_request.split_clusters[d]

                # Update allocate amounts
                self.allocated_cpu[d] -= total_cpu
                self.allocated_memory[d] -= total_memory

                # Update free resources
                self.free_cpu[d] = self.cpu_capacity[d] - self.allocated_cpu[d]
                self.free_memory[d] = self.memory_capacity[d] - self.allocated_memory[d]
        else:
            # logging.info("[Dequeue] Deployment is not split...")
            n = deployment_request.deployed_cluster
            total_cpu = self.deployment_request.num_replicas * self.deployment_request.cpu_request
            total_memory = self.deployment_request.num_replicas * self.deployment_request.memory_request

            # Update allocate amounts
            self.allocated_cpu[n] -= total_cpu
            self.allocated_memory[n] -= total_memory

            # Update free resources
            self.free_cpu[n] = self.cpu_capacity[n] - self.allocated_cpu[n]
            self.free_memory[n] = self.memory_capacity[n] - self.allocated_memory[n]

        # logging.info("[Dequeue] After: ")
        # logging.info("[Dequeue] CPU allocated: {}".format(self.allocated_cpu))
        # logging.info("[Dequeue] CPU free: {}".format(self.free_cpu))
        # logging.info("[Dequeue] MEM allocated: {}".format(self.allocated_memory))
        # logging.info("[Dequeue] MEM free: {}".format(self.free_memory))

    def check_if_cluster_is_really_full(self) -> bool:
        is_full = [self.check_if_cluster_is_full_after_full_deployment(i) for i in range(self.num_clusters)]
        return np.all(is_full)

    def deployment_generator(self):
        deployment_list = get_c2e_deployment_list()
        n = self.np_random.integers(low=0, high=len(deployment_list))
        d = deployment_list[n-1]
        d.num_replicas = self.np_random.integers(low=1, high=MAX_REPLICAS)
        return d

    def next_request(self) -> None:
        arrival_time = self.current_time + self.np_random.exponential(scale=1 / self.arrival_rate_r)
        departure_time = arrival_time + self.np_random.exponential(scale=self.call_duration_r)
        self.dt = departure_time - arrival_time
        self.current_time = arrival_time

        while True:
            if self.running_requests:
                next_departure_time, _ = self.running_requests[0]
                if next_departure_time < arrival_time:
                    self.dequeue_request()
                    continue
            break

        self.deployment_request = self.deployment_generator()
        # logging.info('[Next Request]: Name: {} | Replicas {}'.format(self.deployment_request.name, self.deployment_request.num_replicas))

    '''
    def deployment_list_generator(self, n_deployments: int = 1000) -> None:
        d = {"deployment_requests": []}
        for _ in range(n_deployments):
            d["deployment_requests"].append(self.deployment_generator())
        json_object = json.dumps(d, indent=1)

        with open("./envs/deployments.json", "w") as f:
            f.write(json_object)
    '''
