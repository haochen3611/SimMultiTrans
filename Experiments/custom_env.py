import time
from abc import ABC

from SimMultiTrans import Simulator, Graph, graph_file, vehicle_file, r_name, p_name, RESULTS, Plot
import gym
from gym.spaces import Discrete, Box, MultiDiscrete, Dict, Tuple
import numpy as np
import json

import ray
from ray import tune
from ray.rllib.utils import try_import_tf
from ray.tune import grid_search
tf = try_import_tf()


class TaxiRebalance(gym.Env, ABC):

    def __init__(self, config):
        self._config = config
        self.curr_time = 0
        self.graph = Graph()
        self.graph.import_graph(graph_file)
        self.sim = Simulator(self.graph)
        self.sim.import_arrival_rate(unit=(1, 'sec'))
        self.sim.import_vehicle_attribute(file_name=vehicle_file)
        self.sim.set_running_time(start_time=config['start_time'],
                                  time_horizon=config['time_horizon'],
                                  unit='hour')
        self.sim.routing.set_routing_method(r_name)
        self.sim.rebalance.set_parameters(lazy=config['lazy'], vrange=config['range'])
        self.sim.rebalance.set_policy(p_name)
        self.sim.initialize(seed=0)

        self.total_vehicle = config['total_vehicle']
        self.reb_interval = config['reb_interval']
        self.max_travel_t = config['max_travel_time']
        self.max_lookback_steps = int(np.ceil(self.max_travel_t/self.reb_interval))
        self.max_passenger = config['max_passenger']
        self.num_nodes = len(config['nodes_list'])
        self.near_neighbor = config['near_neighbor']

        self.action_space = Box(low=0, high=1, shape=((self.near_neighbor+1)*self.num_nodes, ), dtype=np.float64)
        self.observation_space = Dict({'p_queue': Box(0, self.max_passenger, shape=(self.num_nodes, ),
                                                      dtype=np.int32),
                                       'v_queue': Box(0, self.total_vehicle, shape=(self.num_nodes, ),
                                                      dtype=np.int32),})
                                       # 'reb_hist': Tuple([MultiDiscrete([self.near_neighbor + 1] * self.num_nodes)]
                                       #                   * self.max_lookback_steps)
                                       # })

        self._is_running = False
        self._done = False
        self._start_time = time.time()
        self._alpha = 0
        self._step = 0
        self._total_steps = self._config["timesteps_total"]
        self._total_vehicle = self.sim.vehicle_attri['taxi']['total']

        self._travel_time = np.zeros((self.num_nodes, self.num_nodes))
        for i, node in enumerate(self.graph.graph_top):
            for j, road in enumerate(self.graph.graph_top):
                if i != j:
                    self._travel_time[i, j] = self.graph.graph_top[node]['node'].road[road].dist
        self._travel_time /= np.linalg.norm(self._travel_time, ord=np.inf)
        self._pre_action = np.zeros((self.near_neighbor+1, self.num_nodes))

    def reset(self):
        if self._is_running:
            self.sim.finishing_touch(self._start_time)
            self.sim.save_result(RESULTS)
            self.sim.plot.combination_queue_animation(mode='taxi', frames=100, autoplay=True)
            self.sim.plot.plot_passenger_queuelen_time(mode='taxi')
            self.sim.plot.plot_passenger_waittime(mode='taxi')
        self.__init__(config=self._config)

        with open(vehicle_file, 'r') as file:
            vehicle_dist = json.load(file)
        vehicle_dist = vehicle_dist['taxi']['distrib']
        vehicle_dist = np.array([vehicle_dist[x] for x in vehicle_dist])
        return dict({'p_queue': np.zeros((self.num_nodes,)), 'v_queue': vehicle_dist})

    def step(self, action):
        assert isinstance(action, np.ndarray)
        self._step += 1
        action = action.reshape((self.near_neighbor+1, self.num_nodes))
        action = action / np.sum(action, axis=1, keepdims=True)
        # action = self._alpha*action + (1 - self._alpha) * np.eye(5)
        # print(action)
        # self._alpha += self._step/self._total_steps

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

        reward = -(p_queue.sum() +
                   np.maximum(np.array(v_queue-p_queue, ndmin=2).T*action*self._travel_time, 0).sum())
        print(reward)
        print('passenger', p_queue)
        print('vehicle', v_queue)
        print(f'at node {v_queue.sum()}, on road {self._total_vehicle - v_queue.sum()}')
        print(f'action diff {np.linalg.norm(self._pre_action-action)}')
        self._pre_action = action
        if self.curr_time >= self._config['time_horizon']*3600 - 1:
            self._done = True
        return dict({'p_queue': p_queue, 'v_queue': v_queue}), reward, self._done, {}


if __name__ == '__main__':

    ray.init()
    with open(graph_file, 'r') as f:
        node_list = json.load(f)
    node_list = [x for x in node_list]
    stop_condition = {
            "timesteps_total": 600,
        }
    analysis = tune.run(
        "SAC",
        stop=stop_condition,
        reuse_actors=True,
        config={
            "env": TaxiRebalance,
            "lr": 1e-4,
            "num_workers": 1,
            "env_config": {
                "start_time": '08:00:00',
                "time_horizon": 100,
                "lazy": 1,
                "range": 20,
                "total_vehicle": 500000,
                "reb_interval": 600,
                "max_travel_time": 1000,
                "max_passenger": 1e6,
                "nodes_list": node_list,
                "near_neighbor": len(node_list)-1,
                "timesteps_total": stop_condition["timesteps_total"]
            }
        }
    )

    print(analysis)
