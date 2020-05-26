import time
from abc import ABC
import os
import sys

from SimMultiTrans import SimpleSimulator, Simulator, Graph, graph_file, vehicle_file
from SimMultiTrans.utils import RESULTS, CONFIG, update_graph_file, update_vehicle_initial_distribution
import gym
from gym.spaces import Discrete, Box, MultiDiscrete, Dict, Tuple
import numpy as np
import json
import argparse as ap
import pandas as pd

import ray
from ray import tune
from ray.rllib.utils import try_import_tf
from ray.tune import grid_search
import ray.rllib.agents.sac as sac
import ray.rllib.agents.dqn as dqn
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print
tf = try_import_tf()
my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices=my_devices, device_type='CPU')


class TaxiRebalance(gym.Env, ABC):

    def __init__(self, config):
        self._config = config
        self._curr_time = 0
        self._graph = Graph()
        self._graph.import_graph(graph_file)
        self._sim = Simulator(self._graph)

        self._max_vehicle = self._config['max_vehicle']
        self._reb_interval = self._config['reb_interval']
        self._max_travel_t = self._config['max_travel_time']
        self._max_lookback_steps = int(np.ceil(self._max_travel_t / self._reb_interval))
        self._max_passenger = self._config['max_passenger']
        self._num_nodes = len(self._config['nodes_list'])
        self._nodes = tuple(self._config['nodes_list'])
        self._num_neighbors = self._config['near_neighbor']
        self._neighbor_map = self._get_neighbors()
        self._dispatch_rate = self._config['dispatch_rate']

        self.action_space = MultiDiscrete([self._num_neighbors+1] * self._num_nodes)
        # self.observation_space = Tuple((Box(0, self._max_passenger, shape=(self._num_nodes,), dtype=np.int64),
        #                                 Box(0, self._max_vehicle, shape=(self._num_nodes,), dtype=np.int64)))
        self.observation_space = Box(-self._max_vehicle, self._max_passenger, shape=(self._num_nodes,), dtype=np.int64)

        self._is_running = False
        self._done = False
        self._start_time = time.time()
        self._alpha = self._config['alpha']
        self._beta = self._config['beta']
        self._step = 0
        self._total_vehicle = None
        self._travel_dist = None
        self._pre_action = None
        self._episode = 0
        self._worker_id = str(hash(time.time()))
        self._save_res_every_ep = int(self._config['save_res_every_ep'])
        self._vehicle_speed = self._config['veh_speed']

    def _get_neighbors(self):
        k = self._config['near_neighbor']
        if k + 1 > len(self._nodes):
            k = len(self._nodes) - 1
            self._num_neighbors = k
        neighbor_map = dict()
        for node in self._nodes:
            dist_lst = [(dest, self._graph.graph_top[node]['nei'][dest]['dist'])
                        for dest in self._graph.graph_top[node]['nei']]
            dist_lst.sort(key=lambda x: x[1])
            neighbor_map[node] = tuple(self._nodes.index(x[0]) for x in dist_lst[:k+1])
        return neighbor_map

    def _preprocess_action(self, action):
        assert isinstance(action, np.ndarray)
        if np.isnan(action).sum() > 0:
            print(self._step)
            action = self.action_space.sample()
        action = np.squeeze(action)
        action_mat = np.zeros((self._num_nodes, self._num_nodes))
        for nd_idx, cnb in enumerate(action):
            nb_idx = self._neighbor_map[self._nodes[nd_idx]][cnb]
            action_mat[nd_idx, nb_idx] = self._dispatch_rate
            if nb_idx != nd_idx:
                action_mat[nd_idx, nd_idx] = 1 - self._dispatch_rate
            else:
                action_mat[nd_idx, nd_idx] = 1
        sim_action = dict()
        for nd_idx, node in enumerate(self._nodes):
            sim_action[node] = action_mat[nd_idx, :]
        return sim_action, action_mat

    def reset(self):
        if self._done:
            self._episode += 1
            self._done = False
            # print(f'Episode: {self._episode} done!')

        if self._is_running:
            self._sim.finishing_touch(self._start_time)
            if self._episode % self._save_res_every_ep == 0:
                self._sim.save_result(RESULTS, self._worker_id, unique_name=False)
                if self._config['plot_queue_len']:
                    self._sim.plot_pass_queue_len(mode='taxi', suffix=self._worker_id)
                    self._sim.plot_pass_wait_time(mode='taxi', suffix=self._worker_id)
            self._is_running = False

        self._curr_time = 0
        self._step = 0

        self._graph = Graph()
        self._graph.import_graph(graph_file)
        self._sim = Simulator(self._graph)
        self._sim.import_arrival_rate(unit=(1, 'sec'))
        self._sim.import_vehicle_attribute(file_name=vehicle_file)
        self._sim.set_running_time(start_time=self._config['start_time'],
                                   time_horizon=self._config['time_horizon'],
                                   unit='hour')
        self._sim.routing.set_routing_method('simplex')
        self._sim.initialize(seed=0)
        self._total_vehicle = self._sim.vehicle_attri['taxi']['total']

        self._travel_dist = np.zeros((self._num_nodes, self._num_nodes))
        for i, node in enumerate(self._graph.graph_top):
            for j, road in enumerate(self._graph.graph_top):
                if i != j:
                    self._travel_dist[i, j] = self._graph.graph_top[node]['node'].road[road].dist
        self._travel_dist /= np.linalg.norm(self._travel_dist, ord=np.inf)
        self._pre_action = np.zeros((self._num_neighbors, self._num_nodes))

        with open(vehicle_file, 'r') as v_file:
            vehicle_dist = json.load(v_file)
        vehicle_dist = vehicle_dist['taxi']['distrib']
        vehicle_dist = np.array([vehicle_dist[x] for x in vehicle_dist])
        return -vehicle_dist

    def step(self, action):
        self._step += 1
        if not self._is_running:
            self._is_running = True
        sim_action, action_mat = self._preprocess_action(action)
        # print(sim_action)
        p_queue, v_queue = self._sim.step(action=sim_action,
                                          step_length=self._reb_interval,
                                          curr_time=self._curr_time)
        self._curr_time += self._reb_interval
        p_queue = np.array(p_queue)
        v_queue = np.array(v_queue)
        reward = -self._beta*(p_queue.sum() * 0 +
                              self._alpha *
                              np.maximum((v_queue-p_queue).reshape((self._num_nodes, 1)) * action_mat *
                                         self._travel_dist, 0).sum())
        # print(self._vehicle_speed)
        # print(reward)
        # print('passenger', p_queue)
        # print('vehicle', v_queue)
        # print(f'at node {v_queue.sum()}, on road {self._total_vehicle - v_queue.sum()}')
        # print(f'action diff {np.linalg.norm(self._pre_action-action)}')
        self._pre_action = action
        if self._curr_time >= self._config['time_horizon']*3600 - 1:
            self._done = True
        return p_queue - v_queue, reward, self._done, {}


class TaxiRebLite(gym.Env, ABC):

    def __init__(self, config):
        self._config = config
        self._curr_time = 0

        self._sim = SimpleSimulator.init_with_configs(graph_config=graph_file,
                                                      vehicle_config=vehicle_file,
                                                      time_horizon=self._config['time_horizon'],
                                                      time_unit='hour')

        self._max_vehicle = self._config['max_vehicle']
        self._reb_interval = self._config['reb_interval']
        self._max_travel_t = self._config['max_travel_time']
        self._max_lookback_steps = int(np.ceil(self._max_travel_t / self._reb_interval))
        self._max_passenger = self._config['max_passenger']
        self._num_nodes = len(self._sim.index_to_node)
        self._nodes = tuple(self._sim.index_to_node)
        self._num_neighbors = self._config['near_neighbor']
        self._neighbor_map = self._get_neighbors()
        self._dispatch_rate = self._config['dispatch_rate']

        self.action_space = MultiDiscrete([self._num_neighbors + 1] * self._num_nodes)
        # self.observation_space = Tuple((Box(0, self._max_passenger, shape=(self._num_nodes,), dtype=np.int64),
        #                                 Box(0, self._max_vehicle, shape=(self._num_nodes,), dtype=np.int64)))
        self.observation_space = Box(-self._max_vehicle, self._max_passenger, shape=(self._num_nodes,), dtype=np.int64)
        self._is_running = False
        self._done = False
        self._start_time = time.time()
        self._alpha = self._config['alpha']
        self._beta = self._config['beta']
        self._step = 0
        self._total_vehicle = None
        self._travel_dist = self._sim.travel_dist_matrix
        self._travel_dist /= np.linalg.norm(self._travel_dist, ord=np.inf)
        self._pre_action = None
        self._episode = 0
        self._worker_id = str(hash(time.time()))
        self._save_res_every_ep = int(self._config['save_res_every_ep'])
        self._vehicle_speed = self._config['veh_speed']

    def _get_neighbors(self):
        k = self._config['near_neighbor']
        if k + 1 > len(self._nodes):
            k = len(self._nodes) - 1
            self._num_neighbors = k
        neighbor_map = dict()
        dist_mat = self._sim.travel_dist_matrix
        for _i, node in enumerate(self._nodes):
            dist_lst = np.argsort(dist_mat[_i, :])
            neighbor_map[node] = tuple(x for x in dist_lst[:k+1])
        return neighbor_map

    def _preprocess_action(self, action):
        assert isinstance(action, np.ndarray)
        if np.isnan(action).sum() > 0:
            print(self._step)
            action = self.action_space.sample()
        action = np.squeeze(action)
        action_mat = np.zeros((self._num_nodes, self._num_nodes))
        for nd_idx, cnb in enumerate(action):
            nb_idx = self._neighbor_map[self._nodes[nd_idx]][cnb]
            if nb_idx != nd_idx:
                action_mat[nd_idx, nb_idx] = self._dispatch_rate
                action_mat[nd_idx, nd_idx] = 1 - self._dispatch_rate
            else:
                action_mat[nd_idx, nd_idx] = 1

        return action_mat

    def reset(self):
        if self._done:
            self._episode += 1
            self._done = False
            # print(f'Episode: {self._episode} done!')
        if self._is_running:
            # self._sim.finishing_touch(self._start_time)
            # if self._episode % self._save_res_every_ep == 0:
            #     self._sim.save_result(RESULTS, self._worker_id, unique_name=False)
            #     if self._config['plot_queue_len']:
            #         self._sim.plot_pass_queue_len(mode='taxi', suffix=self._worker_id)
            #         self._sim.plot_pass_wait_time(mode='taxi', suffix=self._worker_id)
            self._is_running = False
        self._curr_time = 0
        self._step = 0
        p_q_0, v_q_0 = self._sim.reset()

        return p_q_0 - v_q_0

    def step(self, action):
        self._step += 1
        if not self._is_running:
            self._is_running = True
        sim_action = self._preprocess_action(action)
        # print(sim_action)
        p_queue, v_queue = self._sim.step(action=sim_action,
                                          step_length=self._reb_interval)
        self._curr_time += self._reb_interval
        p_queue = np.array(p_queue)
        v_queue = np.array(v_queue)
        reward = -self._beta*(p_queue.sum() * 0 +
                              self._alpha * self._vehicle_speed *
                              np.maximum((v_queue-p_queue).reshape((self._num_nodes, 1)) * sim_action *
                                         self._travel_dist, 0).sum())
        # print(self._vehicle_speed)
        # print(reward)
        # print('passenger', p_queue)
        # print('vehicle', v_queue)
        # print(f'at node {v_queue.sum()}, on road {self._total_vehicle - v_queue.sum()}')
        # print(f'action diff {np.linalg.norm(self._pre_action-action)}')
        self._pre_action = action
        if self._curr_time >= self._config['time_horizon']*3600 - 1:
            self._done = True
        return p_queue - v_queue, reward, self._done, {}


if __name__ == '__main__':

    # unique results directory for every run
    curr_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    RESULTS = os.path.join(RESULTS, curr_time)
    # Config file has priority over CLI arguments
    parser = ap.ArgumentParser(prog="Taxi Rebalance", description="CLI input to Taxi Rebalance")
    parser.add_argument('--config', nargs='?', metavar='<Configuration file path>',
                        type=str, default='None')
    parser.add_argument('--init_veh', nargs='?', metavar='<Initial vehicle per node>', type=int, default=16)
    parser.add_argument('--num_cpu', nargs='?', metavar='<Number of CPU workers>', type=int, default=1)
    parser.add_argument('--num_gpu', nargs='?', metavar='<Number of GPU workers>', type=int, default=0)
    parser.add_argument('--iter', nargs='?', metavar='<Number of iterations>', type=int, default=1)
    parser.add_argument('--lr', nargs='?', metavar='<Learning rate>', type=float, default=5e-3)
    parser.add_argument('--dpr', nargs='?', metavar='<Percentage used for dispatch at each node>',
                        type=float, default=0.9)
    parser.add_argument('--alpha', nargs='?', metavar='<Weight on travel distance>',
                        type=float, default=1)
    parser.add_argument('--beta', nargs='?', metavar='<Reward scaling coefficient>',
                        type=float, default=0.001)
    parser.add_argument('--vf_clip', nargs='?', metavar='<Value function clip parameter>',
                        type=float, default=1000)
    parser.add_argument('--tr_bat_size', nargs='?', metavar='<Training batch size>',
                        type=int, default=4000)
    parser.add_argument('--wkr_smpl_size', nargs='?', metavar='<Worker sample size>',
                        type=int, default=200)
    parser.add_argument('--sgd_bat_size', nargs='?', metavar='<SGD minibatch size>',
                        type=int, default=128)
    parser.add_argument('--veh_speed', nargs='?', metavar='<Average vehicle speed>',
                        type=int, default=10)
    parser.add_argument('--num_neighbor', nargs='?', metavar='<Number of nearest neighbor>',
                        type=int, default=4)
    parser.add_argument('--lite', action='store_true', default=False, help='Use lite version or not')
    parser.add_argument('--no_share', action='store_false', default=True,
                        help='Not share network layers between policy and value function')

    args = parser.parse_args()

    try:
        with open(args.config, 'r') as file:
            file_conf = json.load(file)
    except FileNotFoundError:
        file_conf = None
        if args.config != 'None':
            raise

    # NODES = sorted(pd.read_csv(os.path.join(CONFIG, 'aam.csv'), index_col=0, header=0).index.values.tolist())
    NODES = sorted([236, 237, 186, 170, 141, 162, 140, 238, 142, 229, 239, 48, 161, 107, 263, 262, 234, 68, 100, 143])
    # NODES = sorted([236, 237, 186, 170, 141])
    initial_vehicle = args.init_veh
    iterations = args.iter
    vehicle_speed = args.veh_speed
    if file_conf is not None:
        NODES = sorted(file_conf.pop("nodes", NODES))
        initial_vehicle = int(file_conf.pop("init_veh", initial_vehicle))
        iterations = int(file_conf.pop("iter", iterations))
        vehicle_speed = int(file_conf.pop("veh_speed", vehicle_speed))

    update_graph_file(NODES, os.path.join(CONFIG, 'gps.csv'), os.path.join(CONFIG, 'aam.csv'))
    update_vehicle_initial_distribution(nodes=NODES, veh_dist=[initial_vehicle for i in range(len(NODES))])

    ray.init()
    nodes_list = [str(x) for x in NODES]
    configure = dqn.DEFAULT_CONFIG.copy()
    if not args.lite:
        configure['env'] = TaxiRebalance
    else:
        print('using lite')
        configure['env'] = TaxiRebLite
    configure['num_workers'] = args.num_cpu if args.num_cpu is not None else 1
    configure['num_gpus'] = args.num_gpu if args.num_gpu is not None else 0
    configure['lr'] = args.lr
    configure['env_config'] = {
        "start_time": '08:00:00',
        "time_horizon": 10,  # hours
        "max_vehicle": 500000,
        "reb_interval": 600,  # seconds 60 steps per episode
        "max_travel_time": 1000,
        "max_passenger": 1e6,
        "nodes_list": nodes_list,
        "near_neighbor": args.num_neighbor,
        "plot_queue_len": False,  # do not use plot function for now
        "dispatch_rate": args.dpr,
        "alpha": args.alpha,
        "beta": args.beta,
        "save_res_every_ep": 100,
        "veh_speed": vehicle_speed
    }

    if file_conf is not None:
        env_config = file_conf.pop('env_config', None)
        configure.update(file_conf)
        if env_config is not None:
            configure['env_config'].update(env_config)

    # print(configure['env_config']['veh_speed'])
    # print(configure['env_config']['nodes_list'])
    # print(configure['num_workers'])
    stt = time.time()
    trainer = dqn.DQNTrainer(configure)
    # import cProfile, pstats, io
    # from pstats import SortKey
    # pr = cProfile.Profile()
    # pr.enable()
    for _ in range(iterations):
        print('Iteration:', _+1)
        results = trainer.train()
        if (_+1) % 100 == 0:
            print(pretty_print(results))
    # pr.disable()
    # s = io.StringIO()
    # ps = pstats.Stats(pr, stream=s).sort_stats(SortKey.CUMULATIVE)
    # ps.print_stats()
    # print(s.getvalue())
    check_pt = trainer.save()
    print(f"Model saved at {check_pt}")
    print(time.time()-stt)

    policy = trainer.get_policy()

