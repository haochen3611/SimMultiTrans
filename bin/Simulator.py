#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
from Graph import Graph
from Passenger import Passenger
from Vehicle import Vehicle
from Node import Node
from Routing import Routing
'''
from bin.Control import *
from bin.Network import *
from bin.Plot import Plot

import numpy as np
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
import pandas as pd
# import plotly as pt
# import plotly.express as px
# import plotly.graph_objects as go

import itertools as itt

from IPython.display import HTML

import random
import json
import os
import logging
import multiprocessing
import threading
import copy 
from time import time
from datetime import datetime, timedelta

class Simulator(object):
    def __init__(self, graph):
        self.start_time = 0
        self.time_horizon = 100
        self.time = 0

        self.graph = graph
        self.graph.generate_nodes()

        self.plot = None
        self.routing = None

        self.road_set = {}

        self.multiprocessing_flag = False

        # saved data
        self.passenger_queuelen = {}
        self.vehicle_queuelen = {}
        self.passenger_waittime = {}
        self.total_dist = {
            'total': {},
            'reb': {}
        }

        self.vehicle_attri = {}
        self.vehicel_onroad = []

        for node in self.graph.graph_top:
            self.passenger_queuelen[node] = {}
            self.vehicle_queuelen[node] = {}
            self.passenger_waittime[node] = {}

            self.road_set[node] = self.graph.graph_top[node]['node'].road

        # create results dic
        figs_path = 'results'
        try:
            os.mkdir(figs_path)
        except OSError:
            pass

        try:
            os.remove(f'{figs_path}/Simulator.log')
        except OSError:
            pass

        logging.basicConfig(level = logging.INFO, filename = f'{figs_path}/Simulator.log')
        logging.info('Graph initialized')

    def import_arrival_rate(self, file_name=None, unit=(1,'min')):
        unit_trans = {
            'day': 60*60*24*unit[0],
            'hour': 60*60*unit[0],
            'min': 60*unit[0],
            'sec': 1*unit[0]
        }
        
        (tn1, tn2) = (self.graph.get_allnodes()[0], self.graph.get_allnodes()[1])
        if ('rate' in self.graph.graph_top[tn1]['nei'][tn2].keys()):
            print(f'Rate infomation is embedded in city.json')
            for index, node in enumerate(self.graph.get_allnodes()):
                rate = np.asarray([ self.graph.graph_top[node]['nei'][dest]['rate']/unit_trans[unit[1]] 
                    if (dest != node) else 0 for dest in self.graph.get_allnodes() ])
                self.graph.graph_top[node]['node'].set_arrival_rate( rate )
        elif (file_name == None):
            print('No input data!')
        else:
            print(f'Rate infomation is imported from {file_name}')
            # import from matrix
            file_name = f'{file_name}'
            rate_matrix = (1/unit_trans[unit[1]])*np.loadtxt(file_name, delimiter=',')

            # print('Node: ', self.graph.get_allnodes())
            (row, col) = rate_matrix.shape
            if (row != col) or (row != self.graph.get_size()):
                logging.error('Different dimensions of matrix and nodes')
                print('Error input matirx!')
            else:
                for index, node in enumerate(self.graph.get_allnodes()):
                    rate_matrix[index][index] = 0
                    self.graph.graph_top[node]['node'].set_arrival_rate( rate_matrix[:,index] )
                    # print(self.graph.graph_top[node]['node'].arr_prob_set)
        
        # save graph structure
        file_path = 'results'
        saved_graph = copy.deepcopy(self.graph)
        for node in saved_graph.graph_top:
            saved_graph.graph_top[node]['node'] = None

        with open(f'{file_path}/city_topology.json', 'w') as json_file:
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
            if (mode == 'walk'):
                for node in self.graph.graph_top:
                    v_attri = self.vehicle_attri[mode]
                    vid = 'walk'
                    v = Vehicle(vid=vid, mode=mode, loc=node)
                    v.set_attri(v_attri)
                    self.graph.graph_top[node]['node'].vehicle_arrive(v)
                    self.graph.graph_top[node]['node'].set_walk(v)
            # for others
            else:
                self.total_dist['total'][mode] = 0
                self.total_dist['reb'][mode] = 0

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
                        
                        if (v.type == 'publ'):
                            # public vehicle wait at park
                            self.graph.graph_top[node]['node'].vehicle_park(v, interarrival)
                            interarrival += v.park_time
                        elif (v.type == 'priv'):
                            # private vehicle wait at node
                            self.graph.graph_top[node]['node'].vehicle_arrive(v)

    def set_running_time(self, starttime, timehorizon, unit):
        unit_trans = {
            'day': 60*60*24,
            'hour': 60*60,
            'min': 60,
            'sec': 1
        }
        self.start_time = datetime.strptime(starttime, '%H:%M:%S')
        self.time_horizon = int(timehorizon*unit_trans[unit])

        self.end_time = timedelta(seconds=self.time_horizon) + self.start_time
        # self.start_time = starttime.strftime("%H:%M:%S")
        print(f'Time horizon: {self.time_horizon}')
        print(f'From {self.start_time.strftime("%H:%M:%S")} to {self.end_time.strftime("%H:%M:%S")}')

        # reset data set length
        for node in self.graph.get_allnodes():
            for mode in self.vehicle_attri:
                self.vehicle_queuelen[node][mode] = np.zeros(self.time_horizon)
                self.passenger_queuelen[node][mode] = np.zeros(self.time_horizon)
                self.passenger_waittime[node][mode] = np.zeros(self.time_horizon)


    def ori_dest_generator(self, method):
        if ( method.equal('uniform') ):
            # Generate random passengers
            nodes_set = self.graph.get_allnodes()
            ori = random.choice(nodes_set)
            # print('ori: ',p_ori)
            nodes_set.remove(ori)
            #print(nodes_set)
            dest = random.choice(nodes_set)            
            return (ori, dest)

    def run(self):
        print('Simulation started: ')
        logging.info(f'Simulation started at {time()}')
        start_time = time()

        # list of modes that can rebalance
        reb_list = [ mode for mode in self.vehicle_attri if ( self.vehicle_attri[mode]['reb'] == 'active' ) ]
        reb_flow = {'nodes': self.graph.get_allnodes()}

        # Time horizon
        for timestep in range(self.time_horizon):

            if (self.multiprocessing_flag): 
                
                task = []
                for node in self.graph.get_allnodes():
                    p = threading.Thread(
                        target=self.node_task,
                        args=[self.graph.graph_top[node]['node'], timestep, reb_list, reb_flow]
                    )
                    # p.start()
                    task.append(p)
                for p in task:
                    p.start()
                    p.join()
                '''
                poolarg = [ (self.graph.graph_top[node]['node'], timestep, reb_list, reb_flow) 
                    for node in self.graph.get_allnodes() ]
                with multiprocessing.Pool(processes=os.cpu_count()) as pool:
                    pool.starmap(self.node_task, poolarg)
                '''
            else:
                for node in self.graph.get_allnodes():
                    self.node_task( self.graph.graph_top[node]['node'], timestep, reb_list, reb_flow )
                    '''
                    n = self.graph.graph_top[node]['node']
                    n.syn_time(timestep)

                    for road in self.road_set[node]:
                        n.road[road].syn_time(timestep)
                        n.road[road].leave(self.graph)

                    # n.new_passenger_arrive(self.graph)
                    n.new_passenger_arrive(self.routing)
                    n.match_demands(self.vehicle_attri)
                    
                    # dispatch
                    if ((timestep+1) % 120 == 0):
                        # rebalance for every 300 steps
                        for mode in reb_list:
                            queue_p = [ self.passenger_queuelen[node][mode][timestep-1] for node in self.graph.get_allnodes() ]
                            queue_v = [ self.vehicle_queuelen[node][mode][timestep-1] for node in self.graph.get_allnodes() ]
                            reb_flow[mode] = {}
                            reb_flow[mode]['p'], reb_flow[mode]['reb'] = self.rebalance.Dispatch_active(node=node, mode=mode, queue_p=queue_p, queue_v=queue_v)
                        n.dispatch(reb_flow)
                    
                    # save data
                    for mode in self.vehicle_attri:
                        if (mode in n.mode):
                            self.passenger_queuelen[node][mode][timestep] = len( n.get_passenger_queue(mode) )
                            self.vehicle_queuelen[node][mode][timestep] = len( n.get_vehicle_queue(mode) )

                            self.passenger_waittime[node][mode][timestep] = n.get_average_wait_time(mode)
                            # queuelength_str += 'Time {}: Vel {} queue length: {}'.format(timestep, mode, len( n.get_vehicle_queue(mode) ))
                            # logging.info('Time {}: Vel {} queue length: {}'.format(timestep, mode, qlength))
                    '''
                    
                    # print('{}'.format( len(n.get_vehicle_queue('scooter')) ))
                    # logging.info('Time={}, Node={}: Pas={}'.format(timestep, node, self.passenger_queuelen[node][timestep]))
            
            if (timestep % (self.time_horizon/20) == 0):
                print('-', end='')

        stop_time = time()
        logging.info(f'Simulation ended at {time()}')
        print('\nSimulation ended')
        print(f'Running time: {stop_time-start_time}')

        # print(self.vehicle_attri.keys())
        simulation_info = {
            'Time_horizon': self.time_horizon,
            'Start_time': self.start_time.strftime("%H:%M:%S"),
            'End_time': self.end_time.strftime("%H:%M:%S"),
            'Duration': stop_time-start_time,
            'Vehicle': list(self.vehicle_attri.keys())
        }
        file_path = 'results'
        with open(f'{file_path}/simulation_info.json', 'w') as json_file:
            json.dump(simulation_info, json_file) 

        # logging.info(queuelength_str)
        self.plot = Plot(self.graph, self.time_horizon, self.start_time)

        for node in self.graph.get_allnodes():
            logging.info(f'Node {node} history: {self.passenger_queuelen[node]}')
            # print(self.passenger_waittime[node])

            for mode in self.graph.graph_top[node]['mode'].split(','):
                if (mode != 'walk' and self.vehicle_attri[mode]['reb'] == 'active'):
                    for dest in self.graph.graph_top[node]['node'].road:
                        road = self.graph.graph_top[node]['node'].road[dest]
                        self.total_dist['total'][mode] += road.get_total_distance(mode)
                        self.total_dist['reb'][mode] += road.get_total_reb_distance(mode)
        ''''''
        # print(self.total_dist)
        # print(self.total_reb_dist)
        # self.plot = Plot(self.graph, self.time_horizon)

        # save data to json
        # result_path = 'results'
        # self.save_result(file_path=result_path)

        self.plot.import_queuelength(self.passenger_queuelen, self.vehicle_queuelen)
        self.plot.import_passenger_waittime(self.passenger_waittime)
        # print(self.passenger_waittime)

    def node_task(self, node, timestep, reb_list, reb_flow):
        # n = self.graph.graph_top[node]['node']
        nid = node.id

        # print(nid, timestep)
        node.time = timestep

        for road in self.road_set[nid]:
            node.road[road].time = timestep
            node.road[road].leave(self.graph)

        # n.new_passenger_arrive(self.graph)
        info = {
            'p_queue': self.passenger_queuelen,
            'v_queue': self.vehicle_queuelen,
            'p_wait': self.passegner_waittime,
            'time': timestep
        }
        # print(info)
        self.routing.syn_info(info)
        node.new_passenger_arrive(self.routing)
        node.match_demands(self.vehicle_attri)
        
        # dispatch
        if ((timestep+1) % 120 == 0):
            # rebalance for every 300 steps
            for mode in reb_list:
                queue_p = [ self.passenger_queuelen[node][mode][timestep-1] for node in self.graph.get_allnodes() ]
                queue_v = [ self.vehicle_queuelen[node][mode][timestep-1] for node in self.graph.get_allnodes() ]
                reb_flow[mode] = {}
                reb_flow[mode]['p'], reb_flow[mode]['reb'] = self.rebalance.Dispatch_active(node=nid, mode=mode, queue_p=queue_p, queue_v=queue_v)
            node.dispatch(reb_flow)

        for mode in self.vehicle_attri:
            if (mode in node.mode):
                self.passenger_queuelen[nid][mode][timestep] = len( node.passenger[mode] )
                self.vehicle_queuelen[nid][mode][timestep] = len( node.vehicle[mode] )

                self.passenger_waittime[nid][mode][timestep] = node.get_average_wait_time(mode)
                # queuelength_str += 'Time {}: Vel {} queue length: {}'.format(timestep, mode, len( n.get_vehicle_queue(mode) ))
                # logging.info('Time {}: Vel {} queue length: {}'.format(timestep, mode, qlength))


    def set_multiprocessing(self, flag=False):
        self.multiprocessing_flag = flag

    def save_result(self, path_name):
        try:
            os.mkdir(path_name)
        except OSError:
            # print("Required directories are created.")
            pass

        saved_q_length = copy.deepcopy(self.passenger_queuelen)
        saved_v_length = copy.deepcopy(self.vehicle_queuelen)
        saved_wait_time = copy.deepcopy(self.passenger_waittime)
        saved_total_dist = copy.deepcopy(self.total_dist)

        for node in self.graph.graph_top:
            for mode in self.vehicle_attri:
                saved_q_length[node][mode] = saved_q_length[node][mode].tolist()
                saved_v_length[node][mode] = saved_v_length[node][mode].tolist()
                saved_wait_time[node][mode] = saved_wait_time[node][mode].tolist()

        # print(saved_q_length)
        with open(f'{path_name}/passenger_queue.json', 'w') as json_file:
            json.dump(saved_q_length, json_file)

        with open(f'{path_name}/vehicle_queue.json', 'w') as json_file:
            json.dump(saved_v_length, json_file)

        with open(f'{path_name}/wait_time.json', 'w') as json_file:
            json.dump(saved_wait_time, json_file)

        with open(f'{path_name}/total_distance.json', 'w') as json_file:
            json.dump(saved_total_dist, json_file) 

        del saved_q_length, saved_v_length, saved_wait_time, saved_total_dist


    def plot_topology(self, method='ploty'):
        self.plot.plot_topology(method='plotly')
        
    def plot_passenger_queuelen(self, mode, time):
        self.plot.plot_passenger_queuelen(mode=mode, time=time)

    def passenger_queue_animation(self, mode, frames, autoplay=False, autosave=False, method='plotly'):
        self.plot.passenger_queue_animation(mode, frames, autoplay=autoplay, autosave=autosave, method=method)
        
    def vehicle_queue_animation(self, mode, frames, autoplay=False, autosave=False, method='plotly'):
        self.plot.vehicle_queue_animation(mode, frames, autoplay=autoplay, autosave=autosave, method=method)
        
    def combination_queue_animation(self, mode, frames, autoplay=False, autosave=False, method='plotly'):
        self.plot.combination_queue_animation(mode, frames, autoplay=autoplay, autosave=autosave, method=method)
    
    def passenger_queuelen_time(self, mode, method='plotly'):
        self.plot.plot_passenger_queuelen_time(mode, method=method)

    def passegner_waittime(self, mode, method='plotly'):
        self.plot.plot_passenger_waittime(mode, method=method)


