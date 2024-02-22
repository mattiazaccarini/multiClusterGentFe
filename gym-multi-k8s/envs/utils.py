import csv
from dataclasses import dataclass

import gym
import numpy as np
import numpy.typing as npt

# DeploymentRequest Info
from numpy import mean


@dataclass
class DeploymentRequest:
    name: str
    num_replicas: int
    cpu_request: float
    cpu_limit: float  # limits can be left out
    memory_request: float
    memory_limit: float
    arrival_time: float
    latency_threshold: int  # Latency threshold that should be respected
    departure_time: float
    deployed_cluster: int = None  # All replicas deployed in one cluster
    is_deployment_split: bool = False  # has the deployment request been split?
    split_clusters: [] = None  # what is the distribution of the deployment request?
    expected_latency: int = None  # expected latency after deployment
    expected_cost: int = None  # expected cost after deployment


# NOT used by Karmada Scheduling env!
# Request Info for Fog env
@dataclass
class Request:
    cpu: float
    ram: float
    disk: float
    load: float
    arrival_time: float
    departure_time: float
    latency: int
    serving_node: int = None
    service_type: int = None  # currently: 0==SVE 1==SDP 2==APP 3==LAF


# Reverses a dict
def sort_dict_by_value(d, reverse=False):
    return dict(sorted(d.items(), key=lambda x: x[1], reverse=reverse))


def get_c2e_deployment_list():
    deployment_list = [
        # 1 adapter-amqp
        DeploymentRequest(name="adapter-amqp", num_replicas=1,
                          cpu_request=0.2, cpu_limit=1.0,
                          memory_request=0.3, memory_limit=0.5,
                          arrival_time=0, departure_time=0,
                          latency_threshold=200),
        # 2 adapter-http
        DeploymentRequest(name="adapter-http", num_replicas=1,
                          cpu_request=0.2, cpu_limit=1.0,
                          memory_request=0.3, memory_limit=0.5,
                          arrival_time=0, departure_time=0,
                          latency_threshold=200),
        # 3 adapter-mqtt
        DeploymentRequest(name="adapter-mqtt", num_replicas=1,
                          cpu_request=0.2, cpu_limit=1.0,
                          memory_request=0.3, memory_limit=0.5,
                          arrival_time=0, departure_time=0,
                          latency_threshold=200),
        # 4 adapter-mqtt
        DeploymentRequest(name="artemis", num_replicas=1,
                          cpu_request=0.2, cpu_limit=1.0,
                          memory_request=0.6, memory_limit=0.6,
                          arrival_time=0, departure_time=0,
                          latency_threshold=200),

        # 5 dispatch-router
        DeploymentRequest(name="dispatch-router", num_replicas=1,
                          cpu_request=0.2, cpu_limit=1.0,
                          memory_request=0.06, memory_limit=0.25,
                          arrival_time=0, departure_time=0,
                          latency_threshold=200),

        # 6 ditto-connectivity
        DeploymentRequest(name="ditto-connectivity", num_replicas=1,
                          cpu_request=0.2, cpu_limit=2.0,
                          memory_request=0.7, memory_limit=1.0,
                          arrival_time=0, departure_time=0,
                          latency_threshold=100),

        # 7 ditto-gateway
        DeploymentRequest(name="ditto-gateway", num_replicas=1,
                          cpu_request=0.2, cpu_limit=2.0,
                          memory_request=0.5, memory_limit=0.7,
                          arrival_time=0, departure_time=0,
                          latency_threshold=100),

        # 8 ditto-nginx
        DeploymentRequest(name="ditto-nginx", num_replicas=1,
                          cpu_request=0.05, cpu_limit=0.15,
                          memory_request=0.016, memory_limit=0.032,
                          arrival_time=0, departure_time=0,
                          latency_threshold=100),

        # 9 ditto-policies
        DeploymentRequest(name="ditto-policies", num_replicas=1,
                          cpu_request=0.2, cpu_limit=2.0,
                          memory_request=0.5, memory_limit=0.7,
                          arrival_time=0, departure_time=0,
                          latency_threshold=100),

        # 10 ditto-swagger-ui
        DeploymentRequest(name="ditto-swagger-ui", num_replicas=1,
                          cpu_request=0.05, cpu_limit=0.1,
                          memory_request=0.016, memory_limit=0.032,
                          arrival_time=0, departure_time=0,
                          latency_threshold=400),

        # 11 ditto-things
        DeploymentRequest(name="ditto-things", num_replicas=1,
                          cpu_request=0.2, cpu_limit=2.0,
                          memory_request=0.5, memory_limit=0.7,
                          arrival_time=0, departure_time=0,
                          latency_threshold=200),

        # 12 ditto-things-search
        DeploymentRequest(name="ditto-things-search", num_replicas=1,
                          cpu_request=0.2, cpu_limit=2.0,
                          memory_request=0.5, memory_limit=0.7,
                          arrival_time=0, departure_time=0,
                          latency_threshold=200),

        # 13 ditto-mongo-db
        DeploymentRequest(name="ditto-mongo-db", num_replicas=1,
                          cpu_request=0.2, cpu_limit=2.0,
                          memory_request=0.5, memory_limit=0.7,
                          arrival_time=0, departure_time=0,
                          latency_threshold=200),

        # 14 service-auth
        DeploymentRequest(name="service-auth", num_replicas=1,
                          cpu_request=0.2, cpu_limit=1.0,
                          memory_request=0.2, memory_limit=0.25,
                          arrival_time=0, departure_time=0,
                          latency_threshold=300),

        # 15 service-command-router
        DeploymentRequest(name="service-command-router", num_replicas=1,
                          cpu_request=0.15, cpu_limit=1.0,
                          memory_request=0.2, memory_limit=0.5,
                          arrival_time=0, departure_time=0,
                          latency_threshold=300),

        # 16 service-device-registry
        DeploymentRequest(name="service-device-registry", num_replicas=1,
                          cpu_request=0.2, cpu_limit=1.0,
                          memory_request=0.4, memory_limit=0.4,
                          arrival_time=0, departure_time=0,
                          latency_threshold=200),
    ]
    return deployment_list


def latency_greedy_policy(action_mask: npt.NDArray, lat_val: npt.NDArray, lat_threshold: float) -> int:
    """Returns the index of a feasible node with latency < lat val."""
    # Remove Last Two Actions
    cluster_mask = np.logical_and(action_mask[:-2], lat_val <= lat_threshold)
    feasible_clusters = np.argwhere(cluster_mask == True).flatten()
    # print("Feasible clusters: {}".format(feasible_clusters))

    if len(feasible_clusters) == 0:
        return len(action_mask) - 1
    return np.random.choice(feasible_clusters)
    # return feasible_clusters[np.argmin(lat_val[feasible_clusters])]


def cost_greedy_policy(env: gym.Env, action_mask: npt.NDArray) -> int:
    """Returns the index of the feasible node with the lowest average weighted load
    between cpu, ram, disk and bandwidth. It does not consider QoS latency requirements."""
    feasible_clusters = np.argwhere(action_mask[:-2] == True).flatten()
    # print("Feasible clusters: {}".format(feasible_clusters))

    mean_cost = []
    # Calculate percentage of allocation for CPU and Memory for feasible clusters
    for c in feasible_clusters:
        # print("CPU capacity: {}".format(env.cpu_capacity[n]))
        # print("MEM capacity: {}".format(env.memory_capacity[n]))
        # print("CPU allocated: {}".format(env.allocated_cpu[n]))
        # print("MEM allocated: {}".format(env.allocated_memory[n]))

        type_id = env.cluster_type[c]
        cost = env.default_cluster_types[type_id]['cost']
        mean_cost.append(cost)

    # print("Mean Load (CPU and MEM): {}".format(mean_load))

    if len(feasible_clusters) == 0:
        # print("Return action mask len: {}".format(len(action_mask) - 1))
        return len(action_mask) - 1
    # print("Return: {}".format(feasible_clusters[np.argmin(mean_load)]))
    return feasible_clusters[np.argmin(mean_cost)]


def resource_greedy_policy(env: gym.Env, obs: npt.NDArray, action_mask: npt.NDArray, lat_val: npt.NDArray,
                           lat_threshold: float) -> int:
    """Returns the index of the feasible node with the lowest average load CPU: and memory
    # lower than the latency threshold"""

    # print("Action Mask: {}".format(action_mask))
    # print("Cluster Latency: {}".format(lat_val))
    # print("Request Threshold: {}".format(lat_threshold))

    # Remove Last Two Actions
    # cluster_mask = np.logical_and(action_mask[:-2], lat_val <= lat_threshold)
    # feasible_clusters = np.argwhere(cluster_mask == True).flatten()

    feasible_clusters = np.argwhere(action_mask[:-2] == True).flatten()
    # print("Feasible clusters: {}".format(feasible_clusters))

    mean_load = []
    # Calculate percentage of allocation for CPU and Memory for feasible clusters
    for c in feasible_clusters:
        type_id = env.cluster_type[c]
        cost = env.default_cluster_types[type_id]['cost']
        mean_load.append(cost)
        # cpu = env.allocated_cpu[c] / env.cpu_capacity[c]
        # mem = env.allocated_memory[c] / env.memory_capacity[c]
        # mean_load = (cpu + mem)/2
        # mean_load.append((cpu + mem) / 2)

    # print("Mean Load (CPU and MEM): {}".format(mean_load))

    if len(feasible_clusters) == 0:
        # print("Return action mask len: {}".format(len(action_mask) - 1))
        return len(action_mask) - 1
    # print("Return: {}".format(feasible_clusters[np.argmin(mean_load)]))
    return feasible_clusters[np.argmin(mean_load)]


# TODO: modify function
'''
def save_obs_to_csv(file_name, timestamp, num_pods, desired_replicas, cpu_usage, mem_usage,
                    traffic_in, traffic_out, latency, lstm_1_step, lstm_5_step):
    file = open(file_name, 'a+', newline='')  # append
    # file = open(file_name, 'w', newline='') # new
    with file:
        fields = ['date', 'num_pods', 'cpu', 'mem', 'desired_replicas',
                  'traffic_in', 'traffic_out', 'latency', 'lstm_1_step', 'lstm_5_step']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()  # write header
        writer.writerow(
            {'date': timestamp,
             'num_pods': int("{}".format(num_pods)),
             'cpu': int("{}".format(cpu_usage)),
             'mem': int("{}".format(mem_usage)),
             'desired_replicas': int("{}".format(desired_replicas)),
             'traffic_in': int("{}".format(traffic_in)),
             'traffic_out': int("{}".format(traffic_out)),
             'latency': float("{:.3f}".format(latency)),
             'lstm_1_step': int("{}".format(lstm_1_step)),
             'lstm_5_step': int("{}".format(lstm_5_step))}
        )
'''


def save_to_csv(file_name, episode, reward, ep_block_prob, ep_accepted_requests, avg_latency, avg_cost, execution_time):
    file = open(file_name, 'a+', newline='')  # append
    # file = open(file_name, 'w', newline='')
    with file:
        fields = ['episode', 'reward', 'ep_block_prob', 'ep_accepted_requests', 'avg_latency', 'avg_cost',
                  'execution_time']
        writer = csv.DictWriter(file, fieldnames=fields)
        # writer.writeheader()
        writer.writerow(
            {'episode': episode,
             'reward': float("{:.2f}".format(reward)),
             'ep_block_prob': float("{:.2f}".format(ep_block_prob)),
             'ep_accepted_requests': float("{:.2f}".format(ep_accepted_requests)),
             'avg_latency': float("{:.2f}".format(avg_latency)),
             'avg_cost': float("{:.2f}".format(avg_cost)),
             'execution_time': float("{:.2f}".format(execution_time))}
        )
