#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import json
import logging
import os
import random
from datetime import datetime, timedelta
from time import time

import numpy as np

from SimMultiTrans.bin.Control import *
from SimMultiTrans.bin.Network import *
from SimMultiTrans.bin.Plot import Plot
from SimMultiTrans.utils import RESULTS, CONFIG


logger = logging.getLogger(__name__)


class Simulator(object):

    def __init__(self, graph):
        self.start_time = datetime.today()
        self.end_time = datetime.today()
        self.time_horizon = 100
        self.time = 0

        self.graph = graph
        # self.graph.generate_nodes()  # TODO: Can this be saved?

        self.plot = None
        self.routing = None
        self.rebalance = None

        self.road_set = {}

        self.multiprocessing_flag = False
        self.reb_time = 120
        self._is_running = False
        self._memory = None  # only use in step running mode

        # saved data
        self.passenger_queuelen = {}
        self.vehicle_queuelen = {}
        self.passenger_waittime = {}
        self.total_trip_dist = {
            'total': {},
            'reb': {}
        }
        self.total_trip_time = {
            'total': {},
            'reb': {}
        }
        self.total_trip = {
            'total': {},
            'reb': {}
        }
        self.total_arrival = {
            'total': {},
            'served': {}
        }
        self.not_served = 0

        self.vehicle_attri = {}
        self.vehicle_onroad = []

        for node in self.graph.graph_top:
            self.passenger_queuelen[node] = {}
            self.vehicle_queuelen[node] = {}
            self.passenger_waittime[node] = {}

            self.road_set[node] = self.graph.graph_top[node]['node'].road

        # try:
        #     os.remove(os.path.join(RESULTS, 'Simulator.log'))
        # except OSError:
        #     pass

        logger.info('Graph initialized')

    def import_arrival_rate(self, file_name=None, unit=(1, 'min')):
        unit_trans = {
            'day': 60 * 60 * 24 * unit[0],
            'hour': 60 * 60 * unit[0],
            'min': 60 * unit[0],
            'sec': 1 * unit[0]
        }

        (tn1, tn2) = (self.graph.get_all_nodes()[0], self.graph.get_all_nodes()[1])
        if 'rate' in self.graph.graph_top[tn1]['nei'][tn2].keys():
            logger.debug(f'Rate infomation is embedded in city.json')
            for index, node in enumerate(self.graph.get_all_nodes()):
                rate = np.asarray([self.graph.graph_top[node]['nei'][dest]['rate'] / unit_trans[unit[1]]
                                   if (dest != node) else 0 for dest in self.graph.get_all_nodes()])
                self.graph.graph_top[node]['node'].set_arrival_rate(rate)
        elif file_name is None:
            logger.warning('No input data!')
        else:
            logger.info(f'Rate infomation is imported from {file_name}')
            # import from matrix
            file_name = f'{file_name}'
            rate_matrix = (1 / unit_trans[unit[1]]) * np.loadtxt(file_name, delimiter=',')

            # logger.info('Node: ', self.graph.get_all_nodes())
            (row, col) = rate_matrix.shape
            if (row != col) or (row != self.graph.get_size()):
                logger.error('Different dimensions of matrix and nodes')
                logger.warning('Error input matirx!')
            else:
                for index, node in enumerate(self.graph.get_all_nodes()):
                    rate_matrix[index][index] = 0
                    self.graph.graph_top[node]['node'].set_arrival_rate(rate_matrix[:, index])
                    # logger.info(self.graph.graph_top[node]['node'].arr_prob_set)

    def import_vehicle_attribute(self, file_name):
        with open(f'{file_name}') as file_data:
            self.vehicle_attri = json.load(file_data)
        # total_vehicle = {}
        for mode in self.vehicle_attri:
            self.vehicle_attri[mode]['total'] = 0
            for node in self.vehicle_attri[mode]['distrib']:
                self.vehicle_attri[mode]['total'] += self.vehicle_attri[mode]['distrib'][node]

        # set routing policy and rebalancing policy
        self.routing = Routing(self.graph, self.vehicle_attri)
        self.rebalance = Rebalancing(self.graph, self.vehicle_attri)

    def set_running_time(self, start_time, time_horizon, unit):
        unit_trans = {
            'day': 60 * 60 * 24,
            'hour': 60 * 60,
            'min': 60,
            'sec': 1
        }
        self.start_time = datetime.strptime(start_time, '%H:%M:%S')
        self.time_horizon = int(time_horizon * unit_trans[unit])

        self.end_time = self.start_time + timedelta(seconds=self.time_horizon)
        logger.info(f'Time horizon: {self.time_horizon}')
        logger.info(f'From {self.start_time.strftime("%H:%M:%S")} to {self.end_time.strftime("%H:%M:%S")}')

    def ori_dest_generator(self, method):
        if method.equal('uniform'):
            # Generate random passengers
            nodes_set = self.graph.get_all_nodes()
            ori = random.choice(nodes_set)
            # print('ori: ',p_ori)
            nodes_set.remove(ori)
            # print(nodes_set)
            dest = random.choice(nodes_set)
            return ori, dest

    def initialize(self, seed=0):
        # save graph structure
        saved_graph = copy.deepcopy(self.graph.graph_top)  # TODO: try to avoid this
        for node in saved_graph:
            saved_graph[node]['node'] = None
        with open(os.path.join(RESULTS, 'city_topology.json'), 'w') as json_file:
            json.dump(saved_graph, json_file)
        # print('.', end='')

        # cnt = len(self.graph.get_all_nodes())
        # reset data set length
        for index, node in enumerate(self.graph.get_all_nodes()):
            for mode in self.vehicle_attri:
                self.vehicle_queuelen[node][mode] = np.zeros(self.time_horizon)
                self.passenger_queuelen[node][mode] = np.zeros(self.time_horizon)
                self.passenger_waittime[node][mode] = 0

        # generate passengers
        # st = time()
        total_p = 0
        np.random.seed(seed)
        for index, node in enumerate(self.graph.get_all_nodes()):
            self.graph.graph_top[node]['node'].passenger_generator(time_horizon=self.time_horizon)
            total_p += self.graph.graph_top[node]['node'].total_p
        # print("Total Passengers:", total_p)
        # print('time used', time()-st)

        # generate vehicles
        for mode in self.vehicle_attri:
            # for walk, assign 1 walk to each node initially
            if mode == 'walk':
                for node in self.graph.graph_top:
                    v_attri = self.vehicle_attri[mode]
                    vid = 'walk'
                    v = Vehicle(vid=vid, mode=mode, loc=node)
                    v.set_attri(v_attri)
                    self.graph.graph_top[node]['node'].vehicle_arrive(v)
                    self.graph.graph_top[node]['node'].set_walk(v)
            # for others
            else:
                self.total_trip_dist['total'][mode] = 0
                self.total_trip_dist['reb'][mode] = 0
                self.total_trip_time['total'][mode] = 0
                self.total_trip_time['reb'][mode] = 0

                name_cnt = 0

                cnt = len(self.vehicle_attri[mode]['distrib'])
                # initialize vehilce distribution
                for index, node in enumerate(self.vehicle_attri[mode]['distrib']):
                    interarrival = 0
                    for locv in range(self.vehicle_attri[mode]['distrib'][node]):
                        v_attri = self.vehicle_attri[mode]
                        vid = f'{mode}_{name_cnt}'
                        name_cnt += 1
                        v = Vehicle(vid=vid, mode=mode, loc=node)
                        v.set_attri(v_attri)

                        if v.type == 'publ':
                            # public vehicle wait at park
                            self.graph.graph_top[node]['node'].vehicle_park(v, interarrival)
                            interarrival += v.park_time
                        elif v.type == 'priv':
                            # private vehicle wait at node
                            self.graph.graph_top[node]['node'].vehicle_arrive(v)
        # print('Done!')

    def reset(self):
        raise NotImplementedError

    # @profile
    def step(self, action, step_length, curr_time):
        if not self._is_running:
            logger.info(f'Simulation started at {time()}')
            self._is_running = True
            # reb_list = [mode for mode in self.vehicle_attri if (self.vehicle_attri[mode]['reb'] == 'active')]
            mode = 'taxi'
            reb_flow = dict()
            reb_trans = dict(nodes=self.graph.get_all_nodes())
            # for mode in reb_list:
            reb_flow[mode] = {'p': [], 'reb': False}
            reb_trans[mode] = {'p': [], 'reb': False}
        else:
            reb_flow, reb_trans, mode = self._memory
        queue_p, queue_v = None, None
        for timestep in range(step_length):
            timestep += curr_time
            # match demands first
            for node in self.graph.get_all_nodes():
                self.node_update(node, timestep)
            for node in self.graph.get_all_nodes():
                self.node_match(node, timestep)
            for node in self.graph.get_all_nodes():
                self.node_savedata(node, timestep)

            # rebalancing
            if (timestep + 1) % step_length == 0:
                # for mode in reb_list:
                queue_p = [self.passenger_queuelen[node][mode][timestep] for node in self.graph.get_all_nodes()]
                queue_v = [self.vehicle_queuelen[node][mode][timestep] for node in self.graph.get_all_nodes()]
                reb_flow[mode]['p'], reb_flow[mode]['reb'] = action, True
                # dispatch
                for node in self.graph.get_all_nodes():
                    reb_trans[mode] = {}
                    reb_trans[mode]['reb'] = reb_flow[mode]['reb']
                    if reb_trans[mode]['reb']:
                        reb_trans[mode]['p'] = reb_flow[mode]['p'][node]
                    self.node_rebalance(node, reb_trans)
                for node in self.graph.get_all_nodes():
                    self.node_savedata(node, timestep)
        self._memory = (reb_flow, reb_trans, mode)
        return queue_p, queue_v

    def finishing_touch(self, start_time):
        """
        Only use this at the end when executed by step
        :return:
        """
        self._is_running = False
        # At the end, count all the waiting time of passegners not served
        for node in self.graph.get_all_nodes():
            self.not_served += self.graph.graph_top[node]['node'].passengers_clear()
            for mode in self.vehicle_attri:
                self.passenger_waittime[node][mode] = self.graph.graph_top[node]['node'].get_average_wait_time(mode)

        stop_time = time()
        logger.info(f'Simulation ended at {time()}')
        # print('\nSimulation ended')
        logger.info(f'Running time: {stop_time - start_time}')

        # print(self.vehicle_attri.keys())
        simulation_info = {
            'Time_horizon': self.time_horizon,
            'Start_time': self.start_time.strftime("%H:%M:%S"),
            'End_time': self.end_time.strftime("%H:%M:%S"),
            'Duration': stop_time - start_time,
            'Vehicle': list(self.vehicle_attri.keys()),
            'Routing_method': self.routing.routing_method,
            'Rebalancing_method': self.rebalance.policy
        }
        with open(os.path.join(RESULTS, 'simulation_info.json'), 'w') as json_file:
            json.dump(simulation_info, json_file)

        self.plot = Plot(self.graph, self.time_horizon, self.start_time)

        for node in self.graph.get_all_nodes():
            logger.info(f'Node {node} history: {self.passenger_queuelen[node]}')
            # print(self.passenger_waittime[node])
            self.total_arrival['total'][node] = self.graph.graph_top[node]['node'].total_p
            self.total_arrival['served'][node] = self.graph.graph_top[node]['node'].total_served_p

            self.total_trip['total'][node] = {}
            self.total_trip['reb'][node] = {}

            for mode in self.graph.graph_top[node]['mode'].split(','):
                self.total_trip['total'][node][mode] = 0
                self.total_trip['reb'][node][mode] = 0

                if mode in self.total_trip_dist['total']:
                    for dest in self.graph.graph_top[node]['node'].road:
                        road = self.graph.graph_top[node]['node'].road[dest]
                        self.total_trip_dist['total'][mode] += road.get_total_distance(mode)
                        self.total_trip_time['total'][mode] += road.get_total_time(mode)
                        self.total_trip['total'][node][mode] += road.get_total_trip(mode)

                        if mode in self.total_trip_dist['reb']:
                            self.total_trip_dist['reb'][mode] += road.get_total_reb_distance(mode)
                            self.total_trip_time['reb'][mode] += road.get_total_reb_time(mode)
                            self.total_trip['reb'][node][mode] += road.get_total_reb_trip(mode)

        self.plot.queue_p = self.passenger_queuelen
        self.plot.queue_v = self.vehicle_queuelen
        self.plot.waittime_p = self.passenger_waittime
        # print(self.passenger_waittime)

    def run(self):
        # print('Simulation started: ')
        logger.info(f'Simulation started at {time()}')
        start_time = time()

        # list of modes that can rebalance
        reb_list = [mode for mode in self.vehicle_attri if (self.vehicle_attri[mode]['reb'] == 'active')]
        reb_flow = {}
        reb_trans = {'nodes': self.graph.get_all_nodes()}
        for mode in reb_list:
            reb_flow[mode] = {'p': [], 'reb': False}
            reb_trans[mode] = {'p': [], 'reb': False}

        # reb_flow = {'nodes': self.graph.get_all_nodes()}

        # Time horizon
        for timestep in range(self.time_horizon):
            # reb_flag = False
            # match demands first
            for node in self.graph.get_all_nodes():
                self.node_update(node, timestep)

            for node in self.graph.get_all_nodes():
                self.node_match(node, timestep)

            for node in self.graph.get_all_nodes():
                self.node_savedata(node, timestep)

            # rebalancing
            if (timestep + 1) % self.reb_time == 0:
                # reb_flag = True
                for mode in reb_list:
                    queue_p = [self.passenger_queuelen[node][mode][timestep - 1] for node in self.graph.get_all_nodes()]
                    queue_v = [self.vehicle_queuelen[node][mode][timestep - 1] for node in self.graph.get_all_nodes()]

                    # reb_flow[mode] = {}
                    reb_flow[mode]['p'], reb_flow[mode]['reb'] = self.rebalance.Dispatch_active(mode=mode,
                                                                                                queue_p=queue_p,
                                                                                                queue_v=queue_v)
                    # print(reb_flow[mode]['p'], reb_flow[mode]['reb'])

                # dispatch
                for node in self.graph.get_all_nodes():
                    for mode in reb_list:
                        reb_trans[mode] = {}
                        reb_trans[mode]['reb'] = reb_flow[mode]['reb']
                        if reb_trans[mode]['reb']:
                            reb_trans[mode]['p'] = reb_flow[mode]['p'][node]

                    self.node_rebalance(node, reb_trans)

                for node in self.graph.get_all_nodes():
                    self.node_savedata(node, timestep)

            # if timestep % (self.time_horizon / 20) == 0:
            #     logger.info('-', end='')

        self.finishing_touch(start_time)

    def node_update(self, nid, timestep):
        node = self.graph.graph_top[nid]['node']
        node.time = timestep

        for road in self.road_set[nid]:
            node.road[road].time = timestep
            node.road[road].leave(self.graph)

    def node_match(self, nid, timestep):
        node = self.graph.graph_top[nid]['node']

        info = {
            'p_queue': self.passenger_queuelen,
            'v_queue': self.vehicle_queuelen,
            'p_wait': self.passenger_waittime,
            'time': timestep
        }
        self.routing.syn_info(info)
        node.new_passenger_arrive(self.routing)
        node.match_demands(self.vehicle_attri)

    def node_savedata(self, nid, timestep):
        node = self.graph.graph_top[nid]['node']
        for mode in self.vehicle_attri:
            if mode in node.mode:
                self.passenger_queuelen[nid][mode][timestep] = len(node.passenger[mode])
                self.vehicle_queuelen[nid][mode][timestep] = len(node.vehicle[mode])

    def node_rebalance(self, node, reb_trans):
        # dispatch
        self.graph.graph_top[node]['node'].dispatch(reb_trans)

    def set_multiprocessing(self, flag=False):
        self.multiprocessing_flag = flag

    def save_result(self, path_name, suffix=None, unique_name=True):
        if suffix is None:
            suffix = '_' + str(time()) if unique_name else ''
        else:
            suffix = '_' + str(suffix) + '_' + str(time()) if unique_name else '_' + str(suffix)

        os.makedirs(path_name, exist_ok=True)
        saved_q_length = {}
        saved_v_length = {}
        # saved_wait_time = {}
        for node in self.graph.graph_top:
            saved_q_length[node] = {}
            saved_v_length[node] = {}
            # saved_wait_time[node] = {}
            for mode in self.vehicle_attri:
                saved_q_length[node][mode] = self.passenger_queuelen[node][mode].tolist()
                saved_v_length[node][mode] = self.vehicle_queuelen[node][mode].tolist()
                # saved_wait_time[node][mode] = self.passenger_waittime[node][mode]

        # print(saved_q_length)
        with open(os.path.join(path_name, f'passenger_queue{suffix}.json'), 'w') as json_file:
            json.dump(saved_q_length, json_file)

        with open(os.path.join(path_name, f'vehicle_queue{suffix}.json'), 'w') as json_file:
            json.dump(saved_v_length, json_file)

        with open(os.path.join(path_name, f'wait_time{suffix}.json'), 'w') as json_file:
            json.dump(self.passenger_waittime, json_file)

        total_num_arrival = 0
        for node in self.total_arrival['total']:
            total_num_arrival += self.total_arrival['total'][node]
        saved_metrics = {
            'total_trip': self.total_trip,
            'total_tripdist': self.total_trip_dist,
            'total_triptime': self.total_trip_time,
            'total_arrival': self.total_arrival,
            'total_num_arrival': total_num_arrival,
            'not_served': self.not_served
        }
        with open(os.path.join(path_name, f'metrics{suffix}.json'), 'w') as json_file:
            json.dump(saved_metrics, json_file)

        self.plot.queue_p = self.passenger_queuelen
        self.plot.queue_v = self.vehicle_queuelen
        self.plot.waittime_p = self.passenger_waittime
        self.plot.total_trip = self.total_trip
        self.plot.total_tripdist = self.total_trip_dist
        self.plot.total_triptime = self.total_trip_time
        self.plot.total_arrival = self.total_arrival
        self.plot.sum_totalarrival = total_num_arrival
        self.plot.not_served = self.not_served

        del saved_q_length, saved_v_length

    def plot_topology(self, method='ploty'):
        self.plot.plot_topology(method='plotly')

    def plot_pass_queue_len_at_time(self, mode, time):
        self.plot.plot_passenger_queuelen(mode=mode, time=time)

    def plot_pass_queue_anim(self, mode, frames):
        self.plot.passenger_queue_animation(mode, frames)

    def plot_veh_queue_anim(self, mode, frames):
        self.plot.vehicle_queue_animation(mode, frames)

    def plot_combo_queue_anim(self, mode, frames, suffix: str = None):
        self.plot.combination_queue_animation(mode, frames, suffix)

    def plot_pass_queue_len(self, mode, suffix: str = None):
        self.plot.plot_passenger_queuelen_time(mode, suffix)

    def plot_pass_wait_time(self, mode, suffix: str = None):
        self.plot.plot_passenger_waittime(mode, suffix)
