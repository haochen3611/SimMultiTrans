#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Graph import Graph
from Passenger import Passenger
from Vehicle import Vehicle
from Node import Node

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import itertools as itt

from IPython.display import HTML

import random
import json
import os
import logging
from time import time
from datetime import datetime

class Simulator(object):
    def __init__(self, graph):
        self.time_horizon = 100
        self.time = 0

        self.graph = graph
        self.graph.generate_nodes()

        self.road_set = {}

        # saved data
        self.passenger_queuelen = {}
        self.vehicle_queuelen = {}

        self.vehicel_attri = {}
        self.vehicel_onroad = []

        for node in self.graph.get_graph_dic():
            self.passenger_queuelen[node] = np.zeros(self.time_horizon)
            self.vehicle_queuelen[node] = np.zeros(self.time_horizon)

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

    def import_arrival_rate(self, file_name, unit):
        unit_trans = {
            'day': 60*60*24,
            'hour': 60*60,
            'min': 60,
            'sec': 1
        }
        
        rate_matrix = (1/unit_trans[unit])*np.loadtxt(file_name, delimiter=',')
        print(self.graph.get_allnodes())
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
        with open('{}'.format(file_name)) as file_data:
            self.vehicel_attri = json.load(file_data)
        
        # generate vehicles
        for mode in self.vehicel_attri:
            # self.vehicel[mode] = {}
            name_cnt = 0
            # initialize vehilce distribution
            for node in self.vehicel_attri[mode]['distrib']:
                for locv in range(self.vehicel_attri[mode]['distrib'][node]):
                    v_attri = self.vehicel_attri[mode]
                    vid = '{}{}'.format(mode, name_cnt)
                    name_cnt += 1
                    v = Vehicle(vid=vid, mode=mode, loc=node)
                    v.set_attri(v_attri)
                    self.graph.get_graph_dic()[node]['node'].vehicle_arrive(v)


    def set_running_time(self, timehorizon, unit):
        unit_trans = {
            'day': 60*60*24,
            'hour': 60*60,
            'min': 60,
            'sec': 1
        }
        self.time_horizon = int(timehorizon*unit_trans[unit])
        print('Time steps: {}'.format(self.time_horizon))

        # reset data set length
        for node in self.graph.get_allnodes():
            self.passenger_queuelen[node] = np.zeros(self.time_horizon)
            self.vehicle_queuelen[node] = np.zeros(self.time_horizon)

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

    def start(self):
        print('Simulation started')
        logging.info('Simulation started at {}'.format(time()))
        start_time = time()

        # p_test = Passenger(pid='B1A1',ori='B1',dest='A1',arr_time=0)
        # p_test.get_schdule(self.graph)
        # self.graph.get_graph_dic()['B1']['node'].passenger_arrive(p_test)

        for timestep in range(self.time_horizon):
            for node in self.graph.get_allnodes():
                # print('node=', node)
                n = self.graph.get_graph_dic()[node]['node']
                n.syn_time(timestep)

                for road in self.road_set[node]:
                    n.get_road()[road].syn_time(timestep)
                    n.get_road()[road].leave(self.graph)

                n.new_passenger_arrive(self.graph)
                n.match_demands(self.vehicel_attri)
                self.passenger_queuelen[node][timestep] = len( n.get_passenger_queue() )
                
                # print('{}'.format( len(n.get_vehicle_queue('scooter')) ))
                # logging.info('Time={}, Node={}: Pas={}'.format(timestep, node, self.passenger_queuelen[node][timestep]))
            
            if (timestep % (self.time_horizon/20) == 0):
                print('-', end='')
        stop_time = time()
        logging.info('Simulation ended at {}'.format(time()))
        print('\nSimulation ended')
        print('Running time: ', stop_time-start_time)

        for node in self.graph.get_allnodes():
            logging.info('Node #{}# history: {}'.format(node, self.passenger_queuelen[node]))
        

    def plot_passenger_queuelen(self, time):
        x = [ self.graph.get_node_location(node)[0] for node in self.graph.get_graph_dic() ]
        y = [ self.graph.get_node_location(node)[1] for node in self.graph.get_graph_dic() ]

        fig, ax = self.graph.plot_alledges(x, y)

        # color = np.random.randint(1, 100, size=len(self.get_allnodes()))
        color = [ self.passenger_queuelen[node][time] for node in self.graph.get_graph_dic() ]
        scale = [ 300 if (',' in self.graph.get_graph_dic()[node]['mode']) else 100 for node in self.graph.get_graph_dic() ]

        norm = mpl.colors.Normalize(vmin=0, vmax=self.time_horizon)
        
        plt.scatter(x, y, c=color, s=scale, cmap='Reds', label=color, norm=norm, zorder=2, alpha=0.8, edgecolors='none')
        cbar = plt.colorbar()

        # ax.legend()
        ax.grid(True)
        # plt.legend(loc='lower right', framealpha=1)
        plt.xlabel('Latitude')
        plt.ylabel('Longitude')
        plt.title('Finial Queue Length')

        plt.savefig('results/City_Topology.png', dpi=600)
        print('Plot saved to results/City_Topology.png')


    def animation(self, frames):
        x = [ self.graph.get_node_location(node)[0] for node in self.graph.get_graph_dic() ]
        y = [ self.graph.get_node_location(node)[1] for node in self.graph.get_graph_dic() ]

        fig, ax = self.graph.plot_alledges(x, y)

        color = [ self.passenger_queuelen[node][0] for node in self.graph.get_graph_dic() ]
        scale = [ 300 if (',' in self.graph.get_graph_dic()[node]['mode']) else 100 for node in self.graph.get_graph_dic() ]

        result = np.array([self.passenger_queuelen[node] for node in self.passenger_queuelen])

        norm = mpl.colors.Normalize(vmin=0, vmax=result.max())

        scat = plt.scatter(x, y, c=color, s=scale, cmap='Reds', label=color, norm=norm, zorder=2, alpha=0.8, edgecolors='none')
        cbar = plt.colorbar()

        color_set = np.zeros(shape=(len(self.graph.get_graph_dic()), frames))

        for frame in range(0, int(frames)):
            color_set[:, frame] = [ self.passenger_queuelen[node][ int(frame*self.time_horizon/frames) ] for node in self.graph.get_graph_dic() ]

        # print(color_set)

        def update(frame):
            # color = [ self.passenger_queuelen[node][frame_number*100-1] for node in self.graph.get_graph_dic() ]
            # print(color)
            scat.set_sizes( scale )
            scat.set_array( color_set[:, frame%frames] )
            # print('color update')
            return scat

        # Construct the animation, using the update function as the animation director.
        ani = animation.FuncAnimation(fig=fig, func=update, interval=50, frames=frames, repeat=True)
        '''
        try:
            os.remove('results/animation.log')
        except OSError:
            # logging.warning('Delete log file failed'.format(figs_path))
            pass
        ani.save('results/animation.mp4', fps=12, dpi=300)
        
        with open("results/animation.html", "w") as file_data:
            print(ani.to_html5_video(), file=file_data)
        '''
        # animation.to_html5_video()
        plt.show()
        
