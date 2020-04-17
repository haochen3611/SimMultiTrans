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


# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import plotly as pt
# import plotly.express as px
# import plotly.graph_objects as go


class Simulator(object):

    def __init__(self, graph):
        self.start_time = datetime.today()
        self.end_time = datetime.today()
        self.time_horizon = 100
        self.time = 0

        self.graph = graph
        self.graph.generate_nodes()

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
        self.total_tripdist = {
            'total': {},
            'reb': {}
        }
        self.total_triptime = {
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

        self.vehicle_attri = {}
        self.vehicel_onroad = []

        for node in self.graph.graph_top:
            self.passenger_queuelen[node] = {}
            self.vehicle_queuelen[node] = {}
            self.passenger_waittime[node] = {}

            self.road_set[node] = self.graph.graph_top[node]['node'].road

        try:
            os.remove(os.path.join(RESULTS, 'Simulator.log'))
        except OSError:
            pass

        logging.basicConfig(level=logging.INFO, filename=os.path.join(RESULTS, 'Simulator.log'))
        logging.info('Graph initialized')

    def import_arrival_rate(self, file_name=None, unit=(1, 'min')):
        unit_trans = {
            'day': 60 * 60 * 24 * unit[0],
            'hour': 60 * 60 * unit[0],
            'min': 60 * unit[0],
            'sec': 1 * unit[0]
        }

        (tn1, tn2) = (self.graph.get_allnodes()[0], self.graph.get_allnodes()[1])
        if 'rate' in self.graph.graph_top[tn1]['nei'][tn2].keys():
            print(f'Rate infomation is embedded in city.json')
            for index, node in enumerate(self.graph.get_allnodes()):
                rate = np.asarray([self.graph.graph_top[node]['nei'][dest]['rate'] / unit_trans[unit[1]]
                                   if (dest != node) else 0 for dest in self.graph.get_allnodes()])
                self.graph.graph_top[node]['node'].set_arrival_rate(rate)
        elif file_name is None:
            print('No input data!')
        else:
            print(f'Rate infomation is imported from {file_name}')
            # import from matrix
            file_name = f'{file_name}'
            rate_matrix = (1 / unit_trans[unit[1]]) * np.loadtxt(file_name, delimiter=',')

            # print('Node: ', self.graph.get_allnodes())
            (row, col) = rate_matrix.shape
            if (row != col) or (row != self.graph.get_size()):
                logging.error('Different dimensions of matrix and nodes')
                print('Error input matirx!')
            else:
                for index, node in enumerate(self.graph.get_allnodes()):
                    rate_matrix[index][index] = 0
                    self.graph.graph_top[node]['node'].set_arrival_rate(rate_matrix[:, index])
                    # print(self.graph.graph_top[node]['node'].arr_prob_set)

        # save graph structure
        saved_graph = copy.deepcopy(self.graph)
        for node in saved_graph.graph_top:
            saved_graph.graph_top[node]['node'] = None

        with open(os.path.join(RESULTS, 'city_topology.json'), 'w') as json_file:
            json.dump(saved_graph.graph_top, json_file)
        del saved_graph

    def import_vehicle_attribute(self, file_name):
        with open(f'{file_name}') as file_data:
            self.vehicle_attri = json.load(file_data)
        '''
        # check the input correctness
        mode_list = []
        for node in self.graph.get_allnodes():
            modesinnodes = self.graph.get_graph_dic[node]['node'].mode

        for mode in self.vehicle_attri:
        '''
        # set routing policy and rebalancing policy
        self.routing = Routing(self.graph, self.vehicle_attri)
        self.rebalance = Rebalancing(self.graph, self.vehicle_attri)

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
                self.total_tripdist['total'][mode] = 0
                self.total_tripdist['reb'][mode] = 0
                self.total_triptime['total'][mode] = 0
                self.total_triptime['reb'][mode] = 0

                name_cnt = 0

                # initialize vehilce distribution
                for node in self.vehicle_attri[mode]['distrib']:
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
        print(f'Time horizon: {self.time_horizon}')
        print(f'From {self.start_time.strftime("%H:%M:%S")} to {self.end_time.strftime("%H:%M:%S")}')

        # reset data set length
        for node in self.graph.get_allnodes():
            for mode in self.vehicle_attri:
                self.vehicle_queuelen[node][mode] = np.zeros(self.time_horizon)
                self.passenger_queuelen[node][mode] = np.zeros(self.time_horizon)
                self.passenger_waittime[node][mode] = 0

    def ori_dest_generator(self, method):
        if method.equal('uniform'):
            # Generate random passengers
            nodes_set = self.graph.get_allnodes()
            ori = random.choice(nodes_set)
            # print('ori: ',p_ori)
            nodes_set.remove(ori)
            # print(nodes_set)
            dest = random.choice(nodes_set)
            return ori, dest

    def reset(self):
        raise NotImplementedError

    def step(self, action, step_length, curr_time):
        if not self._is_running:
            logging.info(f'Simulation started at {time()}')
            self._is_running = True
            # reb_list = [mode for mode in self.vehicle_attri if (self.vehicle_attri[mode]['reb'] == 'active')]
            mode = 'taxi'
            reb_flow = dict()
            reb_trans = dict(nodes=self.graph.get_allnodes())
            # for mode in reb_list:
            reb_flow[mode] = {'p': [], 'reb': False}
            reb_trans[mode] = {'p': [], 'reb': False}
        else:
            reb_flow, reb_trans, mode = self._memory
        queue_p, queue_v = None, None
        for timestep in range(step_length):
            timestep += curr_time
            reb_flag = False
            # match demands first
            for node in self.graph.get_allnodes():
                self.node_match(node, timestep)
                # save data for rebalancing
                self.node_savedata(node, timestep)

            # rebalancing
            if (timestep + 1) % step_length == 0:
                reb_flag = True
                # for mode in reb_list:
                queue_p = [self.passenger_queuelen[node][mode][timestep - 1] for node in self.graph.get_allnodes()]
                queue_v = [self.vehicle_queuelen[node][mode][timestep - 1] for node in self.graph.get_allnodes()]
                reb_flow[mode]['p'], reb_flow[mode]['reb'] = action, True
                # dispatch
                for node in self.graph.get_allnodes():
                    reb_trans[mode] = {}
                    reb_trans[mode]['reb'] = reb_flow[mode]['reb']
                    if reb_trans[mode]['reb']:
                        reb_trans[mode]['p'] = reb_flow[mode]['p'][node]
                    self.node_rebalance(node, reb_trans)
                    self.node_savedata(node, timestep)
        self._memory = (reb_flow, reb_trans, mode)
        return queue_p, queue_v

    def finishing_touch(self):
        """
        Only use this at the end when executed by step
        :return:
        """
        self._is_running = False
        # At the end, count all the waiting time of passegners not served
        for node in self.graph.get_allnodes():
            self.graph.graph_top[node]['node'].passengers_clear()
            for mode in self.vehicle_attri:
                self.passenger_waittime[node][mode] = self.graph.graph_top[node]['node'].get_average_wait_time(mode)

        # stop_time = time()
        logging.info(f'Simulation ended at {time()}')
        print('\nSimulation ended')
        # print(f'Running time: {stop_time - start_time}')

        # print(self.vehicle_attri.keys())
        simulation_info = {
            'Time_horizon': self.time_horizon,
            'Start_time': self.start_time.strftime("%H:%M:%S"),
            'End_time': self.end_time.strftime("%H:%M:%S"),
            # 'Duration': stop_time - start_time,
            'Vehicle': list(self.vehicle_attri.keys()),
            'Routing_method': self.routing.routing_method,
            'Rebalancing_method': self.rebalance.policy
        }
        with open(os.path.join(RESULTS, 'simulation_info.json'), 'w') as json_file:
            json.dump(simulation_info, json_file)

        self.plot = Plot(self.graph, self.time_horizon, self.start_time)

        for node in self.graph.get_allnodes():
            logging.info(f'Node {node} history: {self.passenger_queuelen[node]}')
            # print(self.passenger_waittime[node])
            self.total_arrival['total'][node] = self.graph.graph_top[node]['node'].total_p
            self.total_arrival['served'][node] = self.graph.graph_top[node]['node'].total_served_p

            self.total_trip['total'][node] = {}
            self.total_trip['reb'][node] = {}

            for mode in self.graph.graph_top[node]['mode'].split(','):
                self.total_trip['total'][node][mode] = 0
                self.total_trip['reb'][node][mode] = 0

                if mode in self.total_tripdist['total']:
                    for dest in self.graph.graph_top[node]['node'].road:
                        road = self.graph.graph_top[node]['node'].road[dest]
                        self.total_tripdist['total'][mode] += road.get_total_distance(mode)
                        self.total_triptime['total'][mode] += road.get_total_time(mode)
                        self.total_trip['total'][node][mode] += road.get_total_trip(mode)

                        if mode in self.total_tripdist['reb']:
                            self.total_tripdist['reb'][mode] += road.get_total_reb_distance(mode)
                            self.total_triptime['reb'][mode] += road.get_total_reb_time(mode)
                            self.total_trip['reb'][node][mode] += road.get_total_reb_trip(mode)

        self.plot.queue_p = self.passenger_queuelen
        self.plot.queue_v = self.vehicle_queuelen
        self.plot.waittime_p = self.passenger_waittime
        # print(self.passenger_waittime)

    def run(self):
        print('Simulation started: ')
        logging.info(f'Simulation started at {time()}')
        start_time = time()

        # list of modes that can rebalance
        reb_list = [mode for mode in self.vehicle_attri if (self.vehicle_attri[mode]['reb'] == 'active')]
        reb_flow = {}
        reb_trans = {'nodes': self.graph.get_allnodes()}
        for mode in reb_list:
            reb_flow[mode] = {'p': [], 'reb': False}
            reb_trans[mode] = {'p': [], 'reb': False}

        # reb_flow = {'nodes': self.graph.get_allnodes()}

        # Time horizon
        for timestep in range(self.time_horizon):
            reb_flag = False
            # match demands first
            for node in self.graph.get_allnodes():
                if (timestep + 1) % self.reb_time == 0:
                    # reb_trans = {}
                    for mode in reb_list:
                        reb_trans[mode] = {}
                        reb_trans[mode]['reb'] = reb_flow[mode]['reb']
                        if reb_trans[mode]['reb']:
                            reb_trans[mode]['p'] = reb_flow[mode]['p'][node]

                self.node_match(node, timestep)
                # save data for rebalancing
                self.node_savedata(node, timestep)

            # rebalancing
            if (timestep + 1) % self.reb_time == 0:
                reb_flag = True
                for mode in reb_list:
                    queue_p = [self.passenger_queuelen[node][mode][timestep - 1] for node in self.graph.get_allnodes()]
                    queue_v = [self.vehicle_queuelen[node][mode][timestep - 1] for node in self.graph.get_allnodes()]

                    # reb_flow[mode] = {}
                    reb_flow[mode]['p'], reb_flow[mode]['reb'] = self.rebalance.Dispatch_active(mode=mode,
                                                                                                queue_p=queue_p,
                                                                                                queue_v=queue_v)
                    # print(reb_flow[mode]['p'])

                # dispatch
                for node in self.graph.get_allnodes():
                    self.node_rebalance(node, reb_trans)
                    self.node_savedata(node, timestep)

            '''
            if (self.multiprocessing_flag): 
                task = []
                for node in self.graph.get_allnodes():
                    p = threading.Thread(
                        target=self.node_task,
                        args=[self.graph.graph_top[node]['node'], timestep, False, reb_flow]
                    )
                    # p.start()
                    task.append(p)
                for p in task:
                    p.start()
                    p.join()

            else:
                pass
            '''

            if timestep % (self.time_horizon / 20) == 0:
                print('-', end='')

        # At the end, count all the waiting time of passegners not served
        for node in self.graph.get_allnodes():
            self.graph.graph_top[node]['node'].passengers_clear()
            for mode in self.vehicle_attri:
                self.passenger_waittime[node][mode] = self.graph.graph_top[node]['node'].get_average_wait_time(mode)

        stop_time = time()
        logging.info(f'Simulation ended at {time()}')
        print('\nSimulation ended')
        print(f'Running time: {stop_time - start_time}')

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

            # logging.info(queuelength_str)
        self.plot = Plot(self.graph, self.time_horizon, self.start_time)

        for node in self.graph.get_allnodes():
            logging.info(f'Node {node} history: {self.passenger_queuelen[node]}')
            # print(self.passenger_waittime[node])
            self.total_arrival['total'][node] = self.graph.graph_top[node]['node'].total_p
            self.total_arrival['served'][node] = self.graph.graph_top[node]['node'].total_served_p

            self.total_trip['total'][node] = {}
            self.total_trip['reb'][node] = {}

            for mode in self.graph.graph_top[node]['mode'].split(','):
                self.total_trip['total'][node][mode] = 0
                self.total_trip['reb'][node][mode] = 0

                if mode in self.total_tripdist['total']:
                    for dest in self.graph.graph_top[node]['node'].road:
                        road = self.graph.graph_top[node]['node'].road[dest]
                        self.total_tripdist['total'][mode] += road.get_total_distance(mode)
                        self.total_triptime['total'][mode] += road.get_total_time(mode)
                        self.total_trip['total'][node][mode] += road.get_total_trip(mode)

                        if mode in self.total_tripdist['reb']:
                            self.total_tripdist['reb'][mode] += road.get_total_reb_distance(mode)
                            self.total_triptime['reb'][mode] += road.get_total_reb_time(mode)
                            self.total_trip['reb'][node][mode] += road.get_total_reb_trip(mode)

        self.plot.queue_p = self.passenger_queuelen
        self.plot.queue_v = self.vehicle_queuelen
        self.plot.waittime_p = self.passenger_waittime
        # print(self.passenger_waittime)

    def node_match(self, nid, timestep):
        # n = self.graph.graph_top[node]['node']
        node = self.graph.graph_top[nid]['node']

        # print(nid, timestep)
        node.time = timestep

        for road in self.road_set[nid]:
            node.road[road].time = timestep
            node.road[road].leave(self.graph)

        # n.new_passenger_arrive(self.graph)
        info = {
            'p_queue': self.passenger_queuelen,
            'v_queue': self.vehicle_queuelen,
            'p_wait': self.passenger_waittime,
            'time': timestep
        }
        # print(info)
        self.routing.syn_info(info)
        node.new_passenger_arrive(self.routing)
        node.match_demands(self.vehicle_attri)

    def node_savedata(self, nid, timestep):
        node = self.graph.graph_top[nid]['node']
        for mode in self.vehicle_attri:
            if mode in node.mode:
                self.passenger_queuelen[nid][mode][timestep] = len(node.passenger[mode])
                self.vehicle_queuelen[nid][mode][timestep] = len(node.vehicle[mode])

                # self.passenger_waittime[nid][mode] = node.get_average_wait_time(mode)

    def node_rebalance(self, node, reb_trans):
        # dispatch
        self.graph.graph_top[node]['node'].dispatch(reb_trans)

    def set_multiprocessing(self, flag=False):
        self.multiprocessing_flag = flag

    def save_result(self, path_name):
        try:
            os.mkdir(path_name)
        except OSError:
            pass

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
        with open(os.path.join(path_name, 'passenger_queue.json'), 'w') as json_file:
            json.dump(saved_q_length, json_file)

        with open(os.path.join(path_name, 'vehicle_queue.json'), 'w') as json_file:
            json.dump(saved_v_length, json_file)

        with open(os.path.join(path_name, 'wait_time.json'), 'w') as json_file:
            json.dump(self.passenger_waittime, json_file)

        total_num_arrival = 0
        for node in self.total_arrival['total']:
            total_num_arrival += self.total_arrival['total'][node]
        saved_metrics = {
            'total_trip': self.total_trip,
            'total_tripdist': self.total_tripdist,
            'total_triptime': self.total_triptime,
            'total_arrival': self.total_arrival,
            'total_num_arrival': total_num_arrival
        }
        with open(os.path.join(path_name, 'metrics.json'), 'w') as json_file:
            json.dump(saved_metrics, json_file)

        del saved_q_length, saved_v_length

    def plot_topology(self, method='ploty'):
        self.plot.plot_topology(method='plotly')

    def plot_passenger_queuelen(self, mode, time):
        self.plot.plot_passenger_queuelen(mode=mode, time=time)

    def passenger_queue_animation(self, mode, frames, autoplay=False, autosave=False):
        self.plot.passenger_queue_animation(mode, frames, autoplay=autoplay, autosave=autosave)

    def vehicle_queue_animation(self, mode, frames, autoplay=False, autosave=False):
        self.plot.vehicle_queue_animation(mode, frames, autoplay=autoplay, autosave=autosave)

    def combination_queue_animation(self, mode, frames, autoplay=False, autosave=False):
        self.plot.combination_queue_animation(mode, frames, autoplay=autoplay, autosave=autosave)

    def passenger_queuelen_time(self, mode):
        self.plot.plot_passenger_queuelen_time(mode)

    def passenger_waittime(self, mode):
        self.plot.plot_passenger_waittime(mode)
