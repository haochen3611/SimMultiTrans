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

        # saved data
        self.passenger_queuelen = {}
        self.vehicle_queuelen = {}
        self.passenger_waittime = {}

        self.vehicle_attri = {}
        self.vehicel_onroad = []

        for node in self.graph.get_graph_dic():
            self.passenger_queuelen[node] = {}
            self.vehicle_queuelen[node] = {}
            self.passenger_waittime[node] = {}

            self.road_set[node] = self.graph.get_graph_dic()[node]['node'].get_road()

        # create results dic
        figs_path = 'results'
        try:
            os.mkdir(figs_path)
        except OSError:
            # logging.warning('Create result directory {} failed'.format(figs_path))
            pass

        try:
            os.remove('{}/Simulator.log'.format(figs_path))
        except OSError:
            # logging.warning('Delete log file failed'.format(figs_path))
            pass

        logging.basicConfig(level = logging.INFO, filename = '{}/Simulator.log'.format(figs_path))
        logging.info('Graph initialized')

    def import_arrival_rate(self, file_name=None, unit='min'):
        unit_trans = {
            'day': 60*60*24,
            'hour': 60*60,
            'min': 60,
            'sec': 1
        }
        
        (tn1, tn2) = (self.graph.get_allnodes()[0], self.graph.get_allnodes()[1])
        if ('rate' in self.graph.get_graph_dic()[tn1]['nei'][tn2].keys()):
            print('Rate infomation is embedded in the city.json')
            for index, node in enumerate(self.graph.get_allnodes()):
                # rate_matrix[index][index] = 0
                # rate_matrix = []
                rate = np.asarray([ self.graph.get_graph_dic()[node]['nei'][dest]['rate'] if (dest != node) else 0
                    for dest in self.graph.get_allnodes() ])
                # rate[ self.graph.get_allnodes().index(node) ] = 0
                # print(rate)
                self.graph.get_graph_dic()[node]['node'].set_arrival_rate( rate )
        else:
            # import from matrix
            file_name = 'conf/{}'.format(file_name)
            rate_matrix = (1/unit_trans[unit])*np.loadtxt(file_name, delimiter=',')

            # print('Node: ', self.graph.get_allnodes())
            (row, col) = rate_matrix.shape
            if (row != col) or (row != self.graph.get_size()):
                logging.error('Different dimensions of matrix and nodes')
                print('Error input matirx!')
            else:
                for index, node in enumerate(self.graph.get_allnodes()):
                    rate_matrix[index][index] = 0
                    self.graph.get_graph_dic()[node]['node'].set_arrival_rate( rate_matrix[:,index] )
                    # print(self.graph.get_graph_dic()[node]['node'].arr_prob_set)
                
    def import_vehicle_attribute(self, file_name):
        with open('conf/{}'.format(file_name)) as file_data:
            self.vehicle_attri = json.load(file_data)
        '''
        # check the input correctness
        mode_list = []
        for node in self.graph.get_allnodes():
            modesinnodes = self.graph.get_graph_dic[node]['node'].get_mode()

        for mode in self.vehicle_attri:
        ''' 
        self.routing = Routing(self.graph, self.vehicle_attri)

        # generate vehicles
        for mode in self.vehicle_attri:
            # for walk, assign 1 walk to each node initially
            if (mode == 'walk'):
                for node in self.graph.get_graph_dic():
                    v_attri = self.vehicle_attri[mode]
                    vid = 'walk'
                    v = Vehicle(vid=vid, mode=mode, loc=node)
                    v.set_attri(v_attri)
                    self.graph.get_graph_dic()[node]['node'].vehicle_arrive(v)
                    self.graph.get_graph_dic()[node]['node'].set_walk(v)
            # for others
            else:
                name_cnt = 0
            
                # initialize vehilce distribution
                for node in self.vehicle_attri[mode]['distrib']:                
                    interarrival = 0
                    for locv in range(self.vehicle_attri[mode]['distrib'][node]):
                        v_attri = self.vehicle_attri[mode]
                        vid = '{}{}'.format(mode, name_cnt)
                        name_cnt += 1
                        v = Vehicle(vid=vid, mode=mode, loc=node)
                        v.set_attri(v_attri)
                        
                        if (v.get_type() == 'publ'):
                            # public vehicle wait at park
                            self.graph.get_graph_dic()[node]['node'].vehicle_park(v, interarrival)
                            interarrival += v.get_parktime()
                        elif (v.get_type() == 'priv'):
                            # private vehicle wait at node
                            self.graph.get_graph_dic()[node]['node'].vehicle_arrive(v)

    def set_running_time(self, starttime, timehorizon, unit):
        unit_trans = {
            'day': 60*60*24,
            'hour': 60*60,
            'min': 60,
            'sec': 1
        }
        self.start_time = datetime.strptime(starttime, '%H:%M:%S')
        self.time_horizon = int(timehorizon*unit_trans[unit])

        end_time = timedelta(seconds=self.time_horizon) + self.start_time
        print('Time horizon: {}'.format(self.time_horizon))
        print('From {} to {}'.format(starttime, end_time.strftime('%H:%M:%S')))

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
        logging.info('Simulation started at {}'.format(time()))
        start_time = time()

        # list of modes that can rebalance
        self.rebalance = Rebalancing(self.graph, self.vehicle_attri)
        reb_list = [ mode for mode in self.vehicle_attri if ( self.vehicle_attri[mode]['reb'] == 'active' ) ]
        reb_flow = {'nodes': self.graph.get_allnodes()}

        # Time horizon
        for timestep in range(self.time_horizon):

            for node in self.graph.get_allnodes():
                n = self.graph.get_graph_dic()[node]['node']
                n.syn_time(timestep)

                for road in self.road_set[node]:
                    n.get_road()[road].syn_time(timestep)
                    n.get_road()[road].leave(self.graph)

                # n.new_passenger_arrive(self.graph)
                n.new_passenger_arrive(self.routing)
                n.match_demands(self.vehicle_attri)
                
                # dispatch
                if ((timestep+1) % 120 == 0):
                    # rebalance for every 300 steps
                    for mode in reb_list:
                        queue_p = [ self.passenger_queuelen[node][mode][timestep-1]
                            for node in self.graph.get_allnodes() ]
                        queue_v = [ self.vehicle_queuelen[node][mode][timestep-1]
                            for node in self.graph.get_allnodes() ]
                        reb_flow[mode] = {}
                        reb_flow[mode]['p'], reb_flow[mode]['reb'] = self.rebalance.Dispatch_active(node=node, mode=mode, queue_p=queue_p, queue_v=queue_v)
                    n.dispatch(reb_flow)

                # save data
                for mode in self.vehicle_attri:
                    if (mode in n.get_mode()):
                        self.passenger_queuelen[node][mode][timestep] = len( n.get_passenger_queue(mode) )
                        self.vehicle_queuelen[node][mode][timestep] = len( n.get_vehicle_queue(mode) )

                        self.passenger_waittime[node][mode][timestep] = n.get_average_wait_time(mode)
                        # queuelength_str += 'Time {}: Vel {} queue length: {}'.format(timestep, mode, len( n.get_vehicle_queue(mode) ))
                        # logging.info('Time {}: Vel {} queue length: {}'.format(timestep, mode, qlength))
                
                # print('{}'.format( len(n.get_vehicle_queue('scooter')) ))
                # logging.info('Time={}, Node={}: Pas={}'.format(timestep, node, self.passenger_queuelen[node][timestep]))
            
            if (timestep % (self.time_horizon/20) == 0):
                print('-', end='')

        stop_time = time()
        logging.info('Simulation ended at {}'.format(time()))
        print('\nSimulation ended')
        print('Running time: ', stop_time-start_time)

        # logging.info(queuelength_str)
        self.plot = Plot(self.graph, self.time_horizon, self.start_time)
        
        for node in self.graph.get_allnodes():
            logging.info('Node {} history: {}'.format(node, self.passenger_queuelen[node]))
            # print(self.passenger_waittime[node])
        ''''''
        
        # self.plot = Plot(self.graph, self.time_horizon)
        self.plot.import_queuelength(self.passenger_queuelen, self.vehicle_queuelen)
        self.plot.import_passenger_waittime(self.passenger_waittime)

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


