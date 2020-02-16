#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Graph import Graph
from Passenger import Passenger
from Vehicle import Vehicle
from Node import Node

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from IPython.display import HTML

import random
import json
import os
from time import time

class Simulator(object):
    def __init__(self, graph, time_horizon):
        self.graph = None
        self.time_horizon = 0
        self.time = 0

        self.graph = graph
        self.graph.generate_nodes()

        self.time_horizon = time_horizon

        self.passenger_queuelen = {}
        self.vehicle_queuelen = {}

        for node in self.graph.get_graph_dic():
            self.passenger_queuelen[node] = np.zeros(self.time_horizon)
            self.vehicle_queuelen[node] = np.zeros(self.time_horizon)

        # create results dic
        figs_path = 'results'
        try:
            os.mkdir(figs_path)
        except OSError:
            print ("Creation of the directory {} failed".format(figs_path))
        else:
            print ("Successfully created the directory {} ".format(figs_path))
        

    def set_running_time(self, th, unit):
        unit_trans = {
            'day': 60*60*24,
            'hour': 60*60,
            'min': 60,
            'sec': 1
        }
        self.time_horizon = th*unit_trans[unit]

        # reset data set length
        for node in self.graph:
            self.passenger[node] = np.zeros(self.time_horizon)
            self.vehicle[node] = np.zeros(self.time_horizon)

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
        start_time = time()
        for timestep in range(self.time_horizon):
            for node in self.graph.get_allnodes():
                # print('node=', node)
                n = self.graph.get_graph_dic()[node]['node']
                n.syn_time(time)
                n.new_passenger_generator(self.graph)

                self.passenger_queuelen[node][timestep] = len( n.get_passenger_queue() )
            
        stop_time = time()
        print('running time: ', stop_time-start_time)

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
        animation = FuncAnimation(fig=fig, func=update, interval=20, repeat=True, blit=False)
        plt.show()
        
