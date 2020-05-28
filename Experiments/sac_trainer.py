import time
from abc import ABC
import os

from SimMultiTrans import Simulator, Graph, graph_file, vehicle_file
from SimMultiTrans.utils import RESULTS, CONFIG, update_graph_file, update_vehicle_initial_distribution
import gym
from gym.spaces import Discrete, Box, MultiDiscrete, Dict, Tuple
import numpy as np
import json
import argparse as ap

import ray
from ray import tune
from ray.rllib.utils import try_import_tf
from ray.tune import grid_search
import ray.rllib.agents.sac as sac
import ray.rllib.agents.dqn as dqn
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print
tf = try_import_tf()


class TaxiRebalance(gym.Env, ABC):

    def __init__(self, config):
        self._config = config
        self.curr_time = 0
        self.graph = Graph()
        self.graph.import_graph(graph_file)
        self.sim = Simulator(self.graph)

        self.max_vehicle = self._config['max_vehicle']
        self.reb_interval = self._config['reb_interval']
        self.max_travel_t = self._config['max_travel_time']
        self.max_lookback_steps = int(np.ceil(self.max_travel_t/self.reb_interval))
        self.max_passenger = self._config['max_passenger']
        self.num_nodes = len(self._config['nodes_list'])
        self.near_neighbor = self._config['near_neighbor']

        self.action_space = Box(low=0, high=1, shape=((self.near_neighbor+1)*self.num_nodes, ))
        self.observation_space = Tuple((Box(0, self.max_passenger, shape=(self.num_nodes, ), dtype=np.int64),
                                        Box(0, self.max_vehicle, shape=(self.num_nodes, ), dtype=np.int64)))

        self._is_running = False
        self._done = False
        self._start_time = time.time()
        self._alpha = 0
        self._step = 0
        self._total_vehicle = None
        self._travel_time = None
        self._pre_action = None
        self._episode = 0

    def reset(self):
        if self._done:
            self._episode += 1
            self._done = False
            print(f'Episode: {self._episode} done!')

        if self._is_running:
            self.sim.finishing_touch(self._start_time)
            # self.sim.save_result(RESULTS)
            if self._config['plot_queue_len']:
                # self.sim.plot_combo_queue_anim(mode='taxi', frames=100)
                self.sim.plot_pass_queue_len(mode='taxi', suffix=f'ep_{self._episode}')
                self.sim.plot_pass_wait_time(mode='taxi', suffix=f'ep_{self._episode}')
            self._is_running = False

        self.curr_time = 0
        self._alpha = 0
        self._step = 0

        self.graph = Graph()
        self.graph.import_graph(graph_file)
        self.sim = Simulator(self.graph)
        self.sim.import_arrival_rate(unit=(1, 'sec'))
        self.sim.import_vehicle_attribute(file_name=vehicle_file)
        self.sim.set_running_time(start_time=self._config['start_time'],
                                  time_horizon=self._config['time_horizon'],
                                  unit='hour')
        self.sim.routing.set_routing_method('simplex')
        self.sim.initialize(seed=0)
        self._total_vehicle = self.sim.vehicle_attri['taxi']['total']

        self._travel_time = np.zeros((self.num_nodes, self.num_nodes))
        for i, node in enumerate(self.graph.graph_top):
            for j, road in enumerate(self.graph.graph_top):
                if i != j:
                    self._travel_time[i, j] = self.graph.graph_top[node]['node'].road[road].dist
        self._travel_time /= np.linalg.norm(self._travel_time, ord=np.inf)
        self._pre_action = np.zeros((self.near_neighbor + 1, self.num_nodes))

        with open(vehicle_file, 'r') as file:
            vehicle_dist = json.load(file)
        vehicle_dist = vehicle_dist['taxi']['distrib']
        vehicle_dist = np.array([vehicle_dist[x] for x in vehicle_dist])
        return np.zeros((self.num_nodes,)), vehicle_dist

    def step(self, action):
        assert isinstance(action, np.ndarray)
        self._step += 1
        if np.isnan(action).sum() > 0:
            print(self._step)
            action = self.action_space.sample()
        action = action.reshape((self.num_nodes, self.near_neighbor+1))
        action = action / np.sum(action, axis=1, keepdims=True)
        # action = self._alpha*action + (1 - self._alpha) * np.eye(5)
        # print(action)

        if not self._is_running:
            self._is_running = True

        sim_action = dict()
        for idx, node in enumerate(self._config['nodes_list']):
            sim_action[node] = np.squeeze(action[idx, :]/np.sum(action[idx, :]))
        # print(sim_action)
        p_queue, v_queue = self.sim.step(action=sim_action,
                                         step_length=self.reb_interval,
                                         curr_time=self.curr_time)
        self.curr_time += self.reb_interval
        p_queue = np.array(p_queue)
        v_queue = np.array(v_queue)
        reward = -(p_queue.sum() + 16*np.maximum(np.array(v_queue-p_queue, ndmin=2).T*action*self._travel_time, 0).sum())
        # print(reward)
        # print('passenger', p_queue)
        # print('vehicle', v_queue)
        # print(f'at node {v_queue.sum()}, on road {self._total_vehicle - v_queue.sum()}')
        # print(f'action diff {np.linalg.norm(self._pre_action-action)}')
        self._pre_action = action
        if self.curr_time >= self._config['time_horizon']*3600 - 1:
            self._done = True
        return (p_queue, v_queue), reward, self._done, {}


if __name__ == '__main__':

    parser = ap.ArgumentParser(prog="Taxi Rebalance", description="CLI input to Taxi Rebalance")
    parser.add_argument('--config', nargs='?', metavar='<Configuration file path>',
                        type=str, default='None')
    parser.add_argument('--init_veh', nargs='?', metavar='<Initial vehicle per node>', type=int, default=10)
    parser.add_argument('--num_cpu', nargs='?', metavar='<Number of CPU workers>', type=int, default=1)
    parser.add_argument('--num_gpu', nargs='?', metavar='<Number of GPU workers>', type=int, default=0)
    parser.add_argument('--iter', nargs='?', metavar='<Number of iterations>', type=int, default=1)
    parser.add_argument('--a_lr', nargs='?', metavar='<Actor learning rate>', type=float, default=3e-3)
    parser.add_argument('--c_lr', nargs='?', metavar='<Critic learning rate>', type=float, default=3e-3)
    parser.add_argument('--e_lr', nargs='?', metavar='<entropy learning rate>', type=float, default=3e-3)
    args = parser.parse_args()

    # NODES = sorted(pd.read_csv(os.path.join(CONFIG, 'aam.csv'), index_col=0, header=0).index.values.tolist())
    # NODES = sorted([236, 237, 186, 170, 141, 162, 140, 238, 142, 229, 239, 48, 161, 107, 263, 262, 234, 68, 100, 143])
    NODES = sorted([236, 237, 186, 170, 141])

    update_graph_file(NODES, os.path.join(CONFIG, 'gps.csv'), os.path.join(CONFIG, 'aam.csv'))
    update_vehicle_initial_distribution(nodes=NODES, veh_dist=[int(args.init_veh) for i in range(len(NODES))])

    ray.init()
    with open(graph_file, 'r') as f:
        node_list = json.load(f)
    node_list = [x for x in node_list]

    configure = sac.DEFAULT_CONFIG.copy()
    configure['env'] = TaxiRebalance
    try:
        with open(args.config, 'r') as file:
            file_conf = json.load(file)
    except FileNotFoundError:
        configure['num_workers'] = args.num_cpu
        configure['num_gpus'] = args.num_gpu
        configure['timesteps_per_iteration'] = 300  # MDP steps per iteration

        configure["Q_model"] = {
            "hidden_activation": "relu",
            "hidden_layer_sizes": (256, 256)}
        configure["policy_model"] = {
            "hidden_activation": "relu",
            "hidden_layer_sizes": (256, ),
        }
        configure["grad_norm_clipping"] = 10
        configure["tau"] = 1e-3
        configure["target_network_update_freq"] = 30
        configure['optimization'] = {
            "actor_learning_rate": args.a_lr,
            "critic_learning_rate": args.c_lr,
            "entropy_learning_rate": args.e_lr,
        }
        configure['env_config'] = {
                    "start_time": '08:00:00',
                    "time_horizon": 10,  # hours
                    "lazy": 1,
                    "range": 20,
                    "max_vehicle": 500000,
                    "reb_interval": 600,  # seconds 60 steps per episode
                    "max_travel_time": 1000,
                    "max_passenger": 1e6,
                    "nodes_list": node_list,
                    "near_neighbor": len(node_list)-1,
                    "plot_queue_len": False
                }
    else:
        for conf in file_conf:
            configure[conf] = file_conf[conf]
        configure['env_config']['nodes_list'] = node_list
        configure['env_config']["near_neighbor"] = len(node_list)-1

    # with open('sac_0.json', 'w') as file:
    #     json.dump(configure, file, indent=4)

    trainer = sac.SACTrainer(config=configure)
    # trainer = ppo.PPOTrainer(config=configure)
    for _ in range(args.iter):
        # print('Iteration:', i)
        results = trainer.train()
        if _ % 100 == 0:
            print(pretty_print(results))
    check_pt = trainer.save()
    print(f"Model saved at {check_pt}")

    # analysis = tune.run(
    #     "SAC",
    #     stop=stop_condition,
    #     reuse_actors=True,
    #     config={
    #         "env": TaxiRebalance,
    #         "lr": 1e-4,
    #         "num_workers": 1,
    #         "env_config": {
    #             "start_time": '08:00:00',
    #             "time_horizon": 100,
    #             "lazy": 1,
    #             "range": 20,
    #             "total_vehicle": 500000,
    #             "reb_interval": 600,
    #             "max_travel_time": 1000,
    #             "max_passenger": 1e6,
    #             "nodes_list": node_list,
    #             "near_neighbor": len(node_list)-1,
    #             "timesteps_total": stop_condition["timesteps_total"]
    #         }
    #     }
    # )

    # print(analysis)
