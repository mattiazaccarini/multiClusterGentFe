import csv
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt


# DeploymentRequest Info
@dataclass
class DeploymentRequest:
    name: str
    num_replicas: int
    cpu_request: float
    cpu_limit: float  # limits can be left out
    memory_request: float
    memory_limit: float
    arrival_time: float
    departure_time: float
    deployed_cluster: int = None
    split_clusters: [] = None
    is_deployment_split: bool = False


@dataclass
class Cluster:
    num_nodes: int
    available_cpu: float
    available_memory: float
    # cpu_capacity: float
    # memory_capacity: float
    # available_storage: float
    # storage_capacity: float
    # bandwidth_matrix: list[int]
    # latency_matrix: list[int]


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
                          arrival_time=0, departure_time=0),
        # 2 adapter-http
        DeploymentRequest(name="adapter-http", num_replicas=1,
                          cpu_request=0.2, cpu_limit=1.0,
                          memory_request=0.3, memory_limit=0.5,
                          arrival_time=0, departure_time=0),
        # 3 adapter-mqtt
        DeploymentRequest(name="adapter-mqtt", num_replicas=1,
                          cpu_request=0.2, cpu_limit=1.0,
                          memory_request=0.3, memory_limit=0.5,
                          arrival_time=0, departure_time=0),
        # 4 adapter-mqtt
        DeploymentRequest(name="artemis", num_replicas=1,
                          cpu_request=0.2, cpu_limit=1.0,
                          memory_request=0.6, memory_limit=0.6,
                          arrival_time=0, departure_time=0),

        # 5 dispatch-router
        DeploymentRequest(name="dispatch-router", num_replicas=1,
                          cpu_request=0.2, cpu_limit=1.0,
                          memory_request=0.06, memory_limit=0.25,
                          arrival_time=0, departure_time=0),

        # 6 ditto-connectivity
        DeploymentRequest(name="ditto-connectivity", num_replicas=1,
                          cpu_request=0.2, cpu_limit=2.0,
                          memory_request=0.7, memory_limit=1.0,
                          arrival_time=0, departure_time=0),

        # 7 ditto-gateway
        DeploymentRequest(name="ditto-gateway", num_replicas=1,
                          cpu_request=0.2, cpu_limit=2.0,
                          memory_request=0.5, memory_limit=0.7,
                          arrival_time=0, departure_time=0),

        # 8 ditto-nginx
        DeploymentRequest(name="ditto-nginx", num_replicas=1,
                          cpu_request=0.05, cpu_limit=0.15,
                          memory_request=0.016, memory_limit=0.032,
                          arrival_time=0, departure_time=0),

        # 9 ditto-policies
        DeploymentRequest(name="ditto-policies", num_replicas=1,
                          cpu_request=0.2, cpu_limit=2.0,
                          memory_request=0.5, memory_limit=0.7,
                          arrival_time=0, departure_time=0),

        # 10 ditto-swagger-ui
        DeploymentRequest(name="ditto-swagger-ui", num_replicas=1,
                          cpu_request=0.05, cpu_limit=0.1,
                          memory_request=0.016, memory_limit=0.032,
                          arrival_time=0, departure_time=0),

        # 11 ditto-things
        DeploymentRequest(name="ditto-things", num_replicas=1,
                          cpu_request=0.2, cpu_limit=2.0,
                          memory_request=0.5, memory_limit=0.7,
                          arrival_time=0, departure_time=0),

        # 12 ditto-things-search
        DeploymentRequest(name="ditto-things-search", num_replicas=1,
                          cpu_request=0.2, cpu_limit=2.0,
                          memory_request=0.5, memory_limit=0.7,
                          arrival_time=0, departure_time=0),

        # 13 ditto-mongo-db
        DeploymentRequest(name="ditto-mongo-db", num_replicas=1,
                          cpu_request=0.2, cpu_limit=2.0,
                          memory_request=0.5, memory_limit=0.7,
                          arrival_time=0, departure_time=0),

        # 14 service-auth
        DeploymentRequest(name="service-auth", num_replicas=1,
                          cpu_request=0.2, cpu_limit=1.0,
                          memory_request=0.2, memory_limit=0.25,
                          arrival_time=0, departure_time=0),

        # 15 service-command-router
        DeploymentRequest(name="service-command-router", num_replicas=1,
                          cpu_request=0.15, cpu_limit=1.0,
                          memory_request=0.2, memory_limit=0.5,
                          arrival_time=0, departure_time=0),

        # 16 service-device-registry
        DeploymentRequest(name="service-device-registry", num_replicas=1,
                          cpu_request=0.2, cpu_limit=1.0,
                          memory_request=0.4, memory_limit=0.4,
                          arrival_time=0, departure_time=0),
    ]
    return deployment_list


def greedy_qos_policy(action_mask: npt.NDArray, lat_val: npt.NDArray) -> int:
    """Returns the index of the feasible node with the lowest latency.
    This policy does not take into account QoS requirements."""
    feasible_nodes = np.argwhere(action_mask[:-1] == True).flatten()
    if len(feasible_nodes) == 0:
        return len(action_mask)
    return feasible_nodes[np.argmin(lat_val[feasible_nodes])]


def greedy_lb_qos_policy(obs: npt.NDArray, action_mask: npt.NDArray, lat_val: npt.NDArray, service_lat: float) -> int:
    """Returns the index of the feasible node with the lowest average weighted load
    between cpu, ram, disk and bandwidth, and for which the latency is lower than the
    QoS service requirement."""
    node_mask = np.logical_and(action_mask[:-1], lat_val <= service_lat)
    feasible_nodes = np.argwhere(node_mask == True).flatten()
    # cpu, ram, disk and bandwidth are the first 4 columns of the observation matrix
    mean_load = np.mean(obs[feasible_nodes, :4], axis=1).flatten()
    if len(feasible_nodes) == 0:
        return len(action_mask)
    return feasible_nodes[np.argmin(mean_load)]


def greedy_lb_policy(obs: npt.NDArray, action_mask: npt.NDArray) -> int:
    """Returns the index of the feasible node with the lowest average weighted load
    between cpu, ram, disk and bandwidth. It does not consider QoS latency requirements."""
    feasible_nodes = np.argwhere(action_mask[:-1] == True).flatten()
    # cpu, ram, disk and bandwidth are the first 4 columns of the observation matrix
    mean_load = np.mean(obs[feasible_nodes, :4], axis=1).flatten()
    if len(feasible_nodes) == 0:
        return len(action_mask)
    return feasible_nodes[np.argmin(mean_load)]


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


def save_to_csv(file_name, episode, reward, execution_time):
    file = open(file_name, 'a+', newline='')  # append
    # file = open(file_name, 'w', newline='')
    with file:
        fields = ['episode', 'reward', 'execution_time']
        writer = csv.DictWriter(file, fieldnames=fields)
        # writer.writeheader()
        writer.writerow(
            {'episode': episode,
             'reward': float("{:.2f}".format(reward)),
             'execution_time': float("{:.2f}".format(execution_time))}
        )
