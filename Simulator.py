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
            self.passenger_queuelen[node] = {}
            self.vehicle_queuelen[node] = {}

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
        print('Node: ', self.graph.get_allnodes())
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
        '''
        # check the input correctness
        mode_list = []
        for node in self.graph.get_allnodes():
            modesinnodes = self.graph.get_graph_dic[node]['node'].get_mode()

        for mode in self.vehicel_attri:
        ''' 

        # generate vehicles
        for mode in self.vehicel_attri:
            # self.vehicel[mode] = {}
            name_cnt = 0
        
            # initialize vehilce distribution
            for node in self.vehicel_attri[mode]['distrib']:
                interarrival = 0
                for locv in range(self.vehicel_attri[mode]['distrib'][node]):
                    v_attri = self.vehicel_attri[mode]
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


    def set_running_time(self, timehorizon, unit):
        unit_trans = {
            'day': 60*60*24,
            'hour': 60*60,
            'min': 60,
            'sec': 1
        }
        self.time_horizon = int(timehorizon*unit_trans[unit])
        print('Time horizon: {}'.format(self.time_horizon))

        # reset data set length
        for node in self.graph.get_allnodes():
            
            for mode in self.vehicel_attri:
                self.vehicle_queuelen[node][mode] = np.zeros(self.time_horizon)
                self.passenger_queuelen[node][mode] = np.zeros(self.time_horizon)

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

        # queuelength_str = ""

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
                
                # save data
                # self.passenger_queuelen[node][timestep] = len( n.get_passenger_queue() )
                # queuelength_str += 'Time {}: Pas queue length: {}\n'.format(timestep, len( n.get_passenger_queue() ))
                # logging.info('Time {}: Pas queue length: {}'.format(timestep, qlength))
                for mode in self.vehicel_attri:
                    if (mode in n.get_mode()):
                        self.passenger_queuelen[node][mode][timestep] = len( n.get_passenger_queue(mode) )
                        self.vehicle_queuelen[node][mode][timestep] = len( n.get_vehicle_queue(mode) )
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


    def passenger_queue_animation_matplotlib(self, fig, ax, x, y, mode, frames):
        color = [ self.passenger_queuelen[node][mode][0] for node in self.graph.get_graph_dic() ]
        scale = [ 300 if (',' in self.graph.get_graph_dic()[node]['mode']) else 100 for node in self.graph.get_graph_dic() ]

        result = np.array([self.passenger_queuelen[node][mode] for node in self.passenger_queuelen])

        norm = mpl.colors.Normalize(vmin=0, vmax=result.max())

        scat = plt.scatter(x, y, c=color, s=scale, cmap='Reds', label=color, norm=norm, zorder=2, alpha=0.8, edgecolors='none')
        cbar = plt.colorbar()

        color_set = np.zeros(shape=(len(self.graph.get_graph_dic()), frames))

        for frame in range(0, int(frames)):
            color_set[:, frame] = [ self.passenger_queuelen[node][mode][ int(frame*self.time_horizon/frames) ] 
                for node in self.graph.get_graph_dic() ]

        # add another axes at the top left corner of the figure
        axtext = fig.add_axes([0.0,0.95,0.1,0.05])
        # turn the axis labels/spines/ticks off
        axtext.axis("off")
        # place the text to the other axes
        time = axtext.text(0.5,0.5, 'time step={}'.format(0), ha="left", va="top")

        def update(frame):
            scat.set_sizes( scale )
            scat.set_array( color_set[:, frame%frames] )
            time.set_text('time step={}'.format(int(frame*self.time_horizon/frames)))
            return scat,time,

        print('Generate passenger queue ......', end='')
        # Construct the animation, using the update function as the animation director.
        ani = animation.FuncAnimation(fig=fig, func=update, interval=50, frames=frames, repeat=True)


    def passenger_queue_animation(self, mode, frames, autoplay=False, autosave=False, method='matplotlib'):
        x = [ self.graph.get_node_location(node)[0] for node in self.graph.get_graph_dic() ]
        y = [ self.graph.get_node_location(node)[1] for node in self.graph.get_graph_dic() ]

        if (method == 'matplotlib'):
            fig, ax = self.graph.plot_alledges(x, y, method)
            ani = self.passenger_queue_animation_matplotlib(fig=fig, ax=ax, x=x, y=y, mode=mode, frames=frames)
        elif (method == 'plotly'):
            return

        
        file_name = 'results/passenger_queue'
        try:
            os.remove(file_name+'.mp4')
        except OSError:
            # logging.warning('Delete log file failed'.format(figs_path))
            pass
        
        if (autosave):
            ani.save(file_name+'.mp4', fps=12, dpi=300)
        '''
        with open(file_name, "w") as file_data:
            print(ani.to_html5_video(), file=file_data)
        '''
        print('Done')
        # animation.to_html5_video()
        if (autoplay):
            plt.show()

    def vehicle_queue_animation_matplotlib(self, fig, ax, x, y, mode, frames):
        # print(self.vehicle_queuelen['A'][mode])
        color = [ self.vehicle_queuelen[node][mode][0] for node in self.graph.get_graph_dic() ]
        scale = [ 300 if (',' in self.graph.get_graph_dic()[node]['mode']) else 100 for node in self.graph.get_graph_dic() ]

        result = np.array([self.vehicle_queuelen[node][mode] for node in self.vehicle_queuelen])

        norm = mpl.colors.Normalize(vmin=0, vmax=result.max())

        scat = plt.scatter(x, y, c=color, s=scale, cmap='Blues', label=color, norm=norm, zorder=2, alpha=0.8, edgecolors='none')
        cbar = plt.colorbar()

        color_set = np.zeros(shape=(len(self.graph.get_graph_dic()), frames))

        for frame in range(0, int(frames)):
            color_set[:, frame] = [ self.vehicle_queuelen[node][mode][ int(frame*self.time_horizon/frames) ] 
                for node in self.graph.get_graph_dic() ]

        # add another axes at the top left corner of the figure
        axtext = fig.add_axes([0.0,0.95,0.1,0.05])
        # turn the axis labels/spines/ticks off
        axtext.axis("off")
        # place the text to the other axes
        time = axtext.text(0.5,0.5, 'time step={}'.format(0), ha="left", va="top")

        def update(frame):
            scat.set_sizes( scale )
            scat.set_array( color_set[:, frame%frames] )
            time.set_text('time step={}'.format(int(frame*self.time_horizon/frames)))
            return scat,time,

        print('Generate {} queue ......'.format(mode), end='')
        # Construct the animation, using the update function as the animation director.
        ani = animation.FuncAnimation(fig=fig, func=update, interval=50, frames=frames, repeat=True)
        return ani


    def vehicle_queue_animation(self, mode, frames, autoplay=False, autosave=False, method='matplotlib'):
        x = [ self.graph.get_node_location(node)[0] for node in self.graph.get_graph_dic() ]
        y = [ self.graph.get_node_location(node)[1] for node in self.graph.get_graph_dic() ]

        if (method == 'matplotlib'):
            fig, ax = self.graph.plot_alledges(x, y, method)
            ani = self.vehicle_queue_animation_matplotlib(fig=fig, ax=ax, x=x, y=y, mode=mode, frames=frames)
        elif (method == 'plotly'):
            return

        file_name = 'results/{}_queue'.format(mode)
        try:
            os.remove(file_name+'.mp4')
        except OSError:
            # logging.warning('Delete log file failed'.format(figs_path))
            pass

        if (autosave):
            if (method == 'matplotlib'):
                ani.save(file_name+'.mp4', fps=12, dpi=300)
            elif (method == 'plotly'):
                return

        print('Done')

        if (autoplay):
            if (method == 'matplotlib'):
                plt.show()
            elif (method == 'plotly'):
                return
        
        
    def combination_queue_animation_matplotlib(self, fig, ax, x, y, mode, frames):    
        color = [ (self.passenger_queuelen[node][mode][0] - self.vehicle_queuelen[node][mode][0]) 
            for node in self.graph.get_graph_dic() ]
        scale = [ 300 if (',' in self.graph.get_graph_dic()[node]['mode']) else 100 for node in self.graph.get_graph_dic() ]

        result = np.array([ (self.passenger_queuelen[node][mode] - self.vehicle_queuelen[node][mode])
            for node in self.vehicle_queuelen ])

        # norm = mpl.colors.Normalize(vmin=result.min(), vcenter=0, vmax=result.max())
        norm = MidpointNormalize(vmin=result.min(), vcenter=0, vmax=result.max())

        scat = plt.scatter(x, y, c=color, s=scale, cmap='coolwarm', label=color, norm=norm, zorder=2, alpha=0.8, edgecolors='none')
        cbar = plt.colorbar()
        color_set = np.zeros(shape=(len(self.graph.get_graph_dic()), frames))
        for frame in range(0, int(frames)):
            index = int(frame*self.time_horizon/frames)
            color_set[:, frame] = [ (self.passenger_queuelen[node][mode][index] - self.vehicle_queuelen[node][mode][index]) 
                for node in self.graph.get_graph_dic() ]

        # add another axes at the top left corner of the figure
        axtext = fig.add_axes([0.0,0.95,0.1,0.05])
        # turn the axis labels/spines/ticks off
        axtext.axis("off")
        # place the text to the other axes
        time = axtext.text(0.5,0.5, 'time step={}'.format(0), ha="left", va="top")

        def update(frame):
            scat.set_sizes( scale )
            scat.set_array( color_set[:, frame%frames] )
            time.set_text('time step={}'.format(int(frame*self.time_horizon/frames)))
            return scat,time,

        print('Generate passenger and {} queue ......'.format(mode), end='')
        # Construct the animation, using the update function as the animation director.
        ani = animation.FuncAnimation(fig=fig, func=update, interval=300, frames=frames, repeat=True)
        return ani

    def combination_queue_animation(self, mode, frames, autoplay=False, autosave=False, method='matplotlib'):
        x = [ self.graph.get_node_location(node)[0] for node in self.graph.get_graph_dic() ]
        y = [ self.graph.get_node_location(node)[1] for node in self.graph.get_graph_dic() ]

        if (method == 'matplotlib'):
            fig, ax = self.graph.plot_alledges(x, y, method)
            ani = self.combination_queue_animation_matplotlib(fig=fig, ax=ax, x=x, y=y, mode=mode, frames=frames)
        elif (method == 'plotly'):
            return

        fig, ax = self.graph.plot_alledges(x, y)
        # print(self.vehicle_queuelen['A'][mode])
        
        file_name = 'results/{}_combined_queue'.format(mode)
        try:
            os.remove(file_name+'.mp4')
        except OSError:
            # logging.warning('Delete log file failed'.format(figs_path))
            pass
        
        if (autosave):
            if (method == 'matplotlib'):
                ani.save(file_name+'.mp4', fps=12, dpi=300)
            elif (method == 'plotly'):
                return

        print('Done')

        if (autoplay):
            if (method == 'matplotlib'):
                plt.show()
            elif (method == 'plotly'):
                return


class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))