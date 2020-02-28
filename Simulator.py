#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Graph import Graph
from Passenger import Passenger
from Vehicle import Vehicle
from Node import Node
from Routing import Routing
from Converter import MidpointNormalize

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import plotly as pt
import plotly.express as px
import plotly.graph_objects as go

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

        self.routing = None

        self.road_set = {}

        # saved data
        self.passenger_queuelen = {}
        self.vehicle_queuelen = {}

        self.vehicle_attri = {}
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
            # self.vehicel[mode] = {}
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
        print('Simulation started: ')
        logging.info('Simulation started at {}'.format(time()))
        start_time = time()

        # p_test = Passenger(pid='B1A1',ori='B1',dest='A1',arr_time=0)
        # p_test.get_schdule(self.graph)
        # self.graph.get_graph_dic()['B1']['node'].passenger_arrive(p_test)

        # queuelength_str = ''

        for timestep in range(self.time_horizon):
            for node in self.graph.get_allnodes():
                # print('node=', node)
                n = self.graph.get_graph_dic()[node]['node']
                n.syn_time(timestep)

                for road in self.road_set[node]:
                    n.get_road()[road].syn_time(timestep)
                    n.get_road()[road].leave(self.graph)

                # n.new_passenger_arrive(self.graph)
                n.new_passenger_arrive(self.routing)
                n.match_demands(self.vehicle_attri)
                
                # save data
                # self.passenger_queuelen[node][timestep] = len( n.get_passenger_queue() )
                # queuelength_str += 'Time {}: Pas queue length: {}\n'.format(timestep, len( n.get_passenger_queue() ))
                # logging.info('Time {}: Pas queue length: {}'.format(timestep, qlength))
                for mode in self.vehicle_attri:
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
        lat = [ self.graph.get_node_location(node)[0] for node in self.graph.get_graph_dic() ]
        lon = [ self.graph.get_node_location(node)[1] for node in self.graph.get_graph_dic() ]

        fig, ax = self.graph.plot_topology_edges(x, y)
        # color = np.random.randint(1, 100, size=len(self.get_allnodes()))
        color = [ self.passenger_queuelen[node][time] for node in self.graph.get_graph_dic() ]
        scale = [ 300 if (',' in self.graph.get_graph_dic()[node]['mode']) else 100 for node in self.graph.get_graph_dic() ]

        norm = mpl.colors.Normalize(vmin=0, vmax=self.time_horizon)
        
        plt.scatter(x=lon, y=lat, c=color, s=scale, cmap='Reds', label=color, norm=norm, zorder=2, alpha=0.8, edgecolors='none')
        cbar = plt.colorbar()

        # ax.legend()
        ax.grid(True)
        # plt.legend(loc='lower right', framealpha=1)
        plt.xlabel('Latitude')
        plt.ylabel('Longitude')
        plt.title('Finial Queue Length')

        plt.savefig('results/City_Topology.png', dpi=600)
        print('Plot saved to results/City_Topology.png')


    def plotly_sactter_animation_data(self, frames, x, y, color, scale, result):

        fig_dict = {'data': [], 'layout': {}, 'frames': []}

        # fill in most of layout
        fig_dict['layout']['xaxis'] = {'title': 'Latitude'}
        fig_dict['layout']['yaxis'] = {'title': 'Longitude'}
        fig_dict['layout']['hovermode'] = 'closest'
        fig_dict['layout']['sliders'] = {
            'args': [ 'transition', { 'duration': 400, 'easing': 'cubic-in-out' } ],
            'initialValue': '0', 'plotlycommand': 'animate', 'values': range(frames), 'visible': True
        }
        fig_dict['layout']['updatemenus'] = [ {
                'buttons': [ 
                    { 'args': [None, {'frame': {'duration': 500, 'redraw': False},
                      'fromcurrent': True, 'transition': {'duration': 300, 'easing': 'quadratic-in-out'}}],
                      'label': 'Play', 'method': 'animate' },
                    { 'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate', 'transition': {'duration': 0}}],
                      'label': 'Pause', 'method': 'animate' }
                ],
                'direction': 'left', 'pad': {'r': 10, 't': 87}, 
                'showactive': False, 'type': 'buttons',  'x': 0.1, 'xanchor': 'right', 'y': 0, 'yanchor': 'top'
            }  ]

        sliders_dict = { 'active': 0, 'yanchor': 'top', 'xanchor': 'left', 
            'currentvalue': { 'font': {'size': 20}, 'prefix': 'Time:', 'visible': True, 'xanchor': 'right' },
            'transition': {'duration': 300, 'easing': 'cubic-in-out'}, 'pad': {'b': 10, 't': 50},
            'len': 0.9, 'x': 0.1, 'y': 0, 'steps': []  }

        # make data
        time = 0
        # colorsacle = 'OrRd' if (result.min() == 0) else 'balance'
        # set 0 be white
        zp = np.abs(result.min())/(result.max() - result.min())
        # print(scale[:, 0])
        colorsacle = [ [0, '#33691E'], [zp, '#FAFAFA'], [1, '#FF6F00'] ]

        data_dict = { 'x': x, 'y': y, 'mode': 'markers', 'name': 'Queue', 'text': self.graph.get_allnodes(),
            'marker': { 'sizemode': 'area', 'size': scale[:, 0], 'sizeref': 2.*max(scale[:, 0])/(40.**2),
                        'color': color[:, 0], 'colorscale': colorsacle,
                        'cmin': result.min(), 'cmax': result.max(), 'colorbar': dict(title='Queue')  }
        }
        fig_dict['data'].append(data_dict)

        # make frames
        for frame_index in range(frames):
            frame = {'data': [], 'name': str(frame_index)}

            data_dict = { 'x': x, 'y': y, 'mode': 'markers', 'name': 'Queue', 'text': self.graph.get_allnodes(),
            'marker': { 'sizemode': 'area', 'size': scale[:, frame_index], 'sizeref': 2.*max(scale[:, 0])/(40.**2),
                        'color': color[:, frame_index], 'colorscale': colorsacle,
                        'cmin': result.min(), 'cmax': result.max(), 'colorbar': dict(title='Queue')  }
            }
            frame['data'].append(data_dict)

            fig_dict['frames'].append(frame)
            frame_time = timedelta(seconds=frame_index*self.time_horizon/frames) + self.start_time
            slider_step = {'args': [ 
                [frame_index], {'frame': {'duration': 300, 'redraw': False}, 'mode': 'immediate', 'transition': {'duration': 300}} ],
                'label': frame_time.strftime('%H:%M'), 'method': 'animate'}
            sliders_dict['steps'].append(slider_step)

        fig_dict['layout']['sliders'] = [sliders_dict]

        fig = go.Figure(fig_dict)
        fig.update_layout(template='plotly_dark')
        return fig

    def plotly_3d_animation_data(self, frames, data):

        fig_dict = {'data': [], 'layout': {}, 'frames': []}

        # fill in most of layout
        fig_dict['layout']['xaxis'] = {'title': 'Latitude'}
        fig_dict['layout']['yaxis'] = {'title': 'Longitude'}
        fig_dict['layout']['hovermode'] = 'closest'
        fig_dict['layout']['sliders'] = {
            'args': [ 'transition', { 'duration': 400, 'easing': 'cubic-in-out' } ],
            'initialValue': '0', 'plotlycommand': 'animate', 'values': range(frames), 'visible': True
        }
        fig_dict['layout']['updatemenus'] = [ {
                'buttons': [ 
                    { 'args': [None, {'frame': {'duration': 500, 'redraw': False},
                      'fromcurrent': True, 'transition': {'duration': 300, 'easing': 'quadratic-in-out'}}],
                      'label': 'Play', 'method': 'animate' },
                    { 'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate', 'transition': {'duration': 0}}],
                      'label': 'Pause', 'method': 'animate' }
                ],
                'direction': 'left', 'pad': {'r': 10, 't': 87}, 
                'showactive': False, 'type': 'buttons',  'x': 0.1, 'xanchor': 'right', 'y': 0, 'yanchor': 'top'
            }  ]

        sliders_dict = { 'active': 0, 'yanchor': 'top', 'xanchor': 'left', 
            'currentvalue': { 'font': {'size': 20}, 'prefix': 'Time:', 'visible': True, 'xanchor': 'right' },
            'transition': {'duration': 300, 'easing': 'cubic-in-out'}, 'pad': {'b': 10, 't': 50},
            'len': 0.9, 'x': 0.1, 'y': 0, 'steps': []  }

        # make data
        time = 0
        # colorsacle = 'OrRd' if (result.min() == 0) else 'balance'
        # set 0 be white
        # zp = np.abs(result.min())/(result.max() - result.min())
        # print(scale[:, 0])
        # colorsacle = [ [0, '#33691E'], [zp, '#FAFAFA'], [1, '#FF6F00'] ]
        '''
        data_dict = { 'x': x, 'y': y, 'mode': 'markers', 'name': 'Queue', 'text': self.graph.get_allnodes(),
            'marker': { 'sizemode': 'area', 'size': scale[:, 0], 'sizeref': 2.*max(scale[:, 0])/(40.**2),
                        'color': color[:, 0], 'colorscale': colorsacle,
                        'cmin': result.min(), 'cmax': result.max(), 'colorbar': dict(title='Queue')  }
                        }
        '''
        
        # data_dict = data[0]
        fig_dict['data'].append(data[0])

        # make frames
        for frame_index in range(frames):
            frame = {'data': [], 'name': str(frame_index)}
            '''
            data_dict = { 'x': x, 'y': y, 'mode': 'markers', 'name': 'Queue', 'text': self.graph.get_allnodes(),
            'marker': { 'sizemode': 'area', 'size': scale[:, frame_index], 'sizeref': 2.*max(scale[:, 0])/(40.**2),
                        'color': color[:, frame_index], 'colorscale': colorsacle,
                        'cmin': result.min(), 'cmax': result.max(), 'colorbar': dict(title='Queue')  }
            }
            '''
            # print(data[frame_index])
            # frame['data'].append( [data[frame_index]] )
            frame['data'].append( data[frame_index] )

            fig_dict['frames'].append(frame)
            frame_time = timedelta(seconds=frame_index*self.time_horizon/frames) + self.start_time
            slider_step = {'args': [ 
                [frame_index], {'frame': {'duration': 300, 'redraw': False}, 'mode': 'immediate', 'transition': {'duration': 300}} ],
                'label': frame_time.strftime('%H:%M'), 'method': 'animate'}
            sliders_dict['steps'].append(slider_step)

        fig_dict['layout']['sliders'] = [sliders_dict]

        fig = go.Figure(fig_dict)
        fig.update_layout(template='plotly_dark')
        return fig

        
    def passenger_queue_animation_matplotlib(self, fig, x, y, mode, frames):
        color = [ self.passenger_queuelen[node][mode][0] for node in self.graph.get_graph_dic() ]
        scale = [ 300 if (',' in self.graph.get_graph_dic()[node]['mode']) else 100 for node in self.graph.get_graph_dic() ]

        result = np.array([self.passenger_queuelen[node][mode] for node in self.passenger_queuelen])

        norm = mpl.colors.Normalize(vmin=0, vmax=result.max())

        scat = plt.scatter(x, y, c=color, s=scale, cmap='Reds', label=color, norm=norm, zorder=2, alpha=0.8, edgecolors='none')
        cbar = plt.colorbar()

        color_set = np.zeros(shape=(len(self.graph.get_graph_dic()), frames))
        scale_set = np.zeros(shape=(len(self.graph.get_graph_dic()), frames))
        for frame in range(0, int(frames)):
            color_set[:, frame] = [ self.passenger_queuelen[node][mode][ int(frame*self.time_horizon/frames) ] 
                for node in self.graph.get_graph_dic() ]
            scale_set[:, frame] = [ self.passenger_queuelen[node][mode][ int(frame*self.time_horizon/frames) ] +100
                for node in self.graph.get_graph_dic() ]
            

        # add another axes at the top left corner of the figure
        axtext = fig.add_axes([0.0,0.95,0.1,0.05])
        # turn the axis labels/spines/ticks off
        axtext.axis('off')
        # place the text to the other axes
        time = axtext.text(0.5,0.5, 'time step={}'.format(0), ha='left', va='top')

        def update(frame):
            scat.set_sizes( scale_set[:, frame%frames] )
            scat.set_array( color_set[:, frame%frames] )
            time.set_text('time step={}'.format(int(frame*self.time_horizon/frames)))
            return scat,time,

        print('Generate Passenger queue ......'.format(mode), end='')
        # Construct the animation, using the update function as the animation director.
        ani = animation.FuncAnimation(fig=fig, func=update, interval=50, frames=frames, repeat=True)
        return ani


    def passenger_queue_animation_plotly(self, fig, x, y, mode, frames):
        
        color = [ self.passenger_queuelen[node][mode][0] for node in self.graph.get_graph_dic() ]
        scale = [ 60 if (',' in self.graph.get_graph_dic()[node]['mode']) else 30 for node in self.graph.get_graph_dic() ]

        result = np.array([self.passenger_queuelen[node][mode] for node in self.passenger_queuelen])

        color_set = np.zeros(shape=(len(self.graph.get_graph_dic()), frames))
        scale_set = np.zeros(shape=(len(self.graph.get_graph_dic()), frames))
        for frame_index in range(0, int(frames)):
            color_set[:, frame_index] = [ self.passenger_queuelen[node][mode][ int(frame_index*self.time_horizon/frames) ] 
                for node in self.graph.get_graph_dic() ]
            scale_set[:, frame_index] = [ self.passenger_queuelen[node][mode][ int(frame_index*self.time_horizon/frames) ] +100
                for node in self.graph.get_graph_dic() ]

        fig = self.plotly_sactter_animation_data(frames=frames, x=x, y=y, color=color_set, scale=scale_set, result=result)
        
        return fig


    def passenger_queue_animation(self, mode, frames, autoplay=False, autosave=False, method='matplotlib'):
        lat = [ self.graph.get_node_location(node)[0] for node in self.graph.get_graph_dic() ]
        lon = [ self.graph.get_node_location(node)[1] for node in self.graph.get_graph_dic() ]

        fig = self.graph.plot_topology_edges(lon, lat, method)
        if (method == 'matplotlib'):
            ani = self.passenger_queue_animation_matplotlib(fig=fig, x=lon, y=lat, mode=mode, frames=frames)
        elif (method == 'plotly'):
            ani = self.passenger_queue_animation_plotly(fig=fig, x=lon, y=lat, mode=mode, frames=frames)
        
        file_name = 'results/passenger_queue'
        try:
            os.remove(file_name+'.mp4')
        except OSError:
            # logging.warning('Delete log file failed'.format(figs_path))
            pass
        
        if (autosave):
            ani.save(file_name+'.mp4', fps=12, dpi=300)

        print('Done')
        # animation.to_html5_video()
        if (autoplay):
            if (method == 'matplotlib'):
                plt.show()
            elif (method == 'plotly'):
                pt.offline.plot(ani, filename=file_name+'.html')


    def vehicle_queue_animation_matplotlib(self, fig, x, y, mode, frames):
        # print(self.vehicle_queuelen['A'][mode])
        color = [ self.vehicle_queuelen[node][mode][0] for node in self.graph.get_graph_dic() ]
        scale = [ 300 if (',' in self.graph.get_graph_dic()[node]['mode']) else 100 for node in self.graph.get_graph_dic() ]

        result = np.array([self.vehicle_queuelen[node][mode] for node in self.vehicle_queuelen])

        norm = mpl.colors.Normalize(vmin=0, vmax=result.max())

        scat = plt.scatter(x, y, c=color, s=scale, cmap='Blues', label=color, norm=norm, zorder=2, alpha=0.8, edgecolors='none')
        cbar = plt.colorbar()

        color_set = np.zeros(shape=(len(self.graph.get_graph_dic()), frames))
        for frame_index in range(0, int(frames)):
            color_set[:, frame_index] = [ self.vehicle_queuelen[node][mode][ int(frame_index*self.time_horizon/frames) ] 
                for node in self.graph.get_graph_dic() ]

        # add another axes at the top left corner of the figure
        axtext = fig.add_axes([0.0,0.95,0.1,0.05])
        # turn the axis labels/spines/ticks off
        axtext.axis('off')
        # place the text to the other axes
        time = axtext.text(0.5,0.5, 'time step={}'.format(0), ha='left', va='top')

        def update(frame):
            scat.set_sizes( scale )
            scat.set_array( color_set[:, frame%frames] )
            time.set_text('time step={}'.format(int(frame*self.time_horizon/frames)))
            return scat,time,

        print('Generate {} queue ......'.format(mode), end='')
        # Construct the animation, using the update function as the animation director.
        ani = animation.FuncAnimation(fig=fig, func=update, interval=50, frames=frames, repeat=True)
        return ani


    def vehicle_queue_animation_plotly(self, fig, x, y, mode, frames):
        
        color = [ self.vehicle_queuelen[node][mode][0] for node in self.graph.get_graph_dic() ]
        scale = [ 300 if (',' in self.graph.get_graph_dic()[node]['mode']) else 100 for node in self.graph.get_graph_dic() ]

        result = np.array([self.vehicle_queuelen[node][mode] for node in self.vehicle_queuelen])

        color_set = np.zeros(shape=(len(self.graph.get_graph_dic()), frames))
        scale_set = np.zeros(shape=(len(self.graph.get_graph_dic()), frames))
        for frame_index in range(0, int(frames)):
            color_set[:, frame_index] = [ self.vehicle_queuelen[node][mode][ int(frame_index*self.time_horizon/frames) ] 
                for node in self.graph.get_graph_dic() ]
            scale_set[:, frame_index] = [ self.vehicle_queuelen[node][mode][ int(frame_index*self.time_horizon/frames) ] +100
                for node in self.graph.get_graph_dic() ]

        fig = self.plotly_sactter_animation_data(frames=frames, x=x, y=y, color=color_set, scale=scale_set, result=result)
        
        return fig


    def vehicle_queue_animation(self, mode, frames, autoplay=False, autosave=False, method='matplotlib'):
        lat = [ self.graph.get_node_location(node)[0] for node in self.graph.get_graph_dic() ]
        lon = [ self.graph.get_node_location(node)[1] for node in self.graph.get_graph_dic() ]

        fig = self.graph.plot_topology_edges(lon, lat, method)

        if (method == 'matplotlib'):
            ani = self.vehicle_queue_animation_matplotlib(fig=fig, x=lon, y=lat, mode=mode, frames=frames)
        elif (method == 'plotly'):
            ani = self.vehicle_queue_animation_plotly(fig=fig, x=lon, y=lat, mode=mode, frames=frames)

        file_name = 'results/{}_queue'.format(mode)
        try:
            os.remove(file_name+'.mp4')
        except OSError:
            # logging.warning('Delete log file failed'.format(figs_path))
            pass

        if (autosave):
            ani.save(file_name+'.mp4', fps=12, dpi=300)

        print('Done')
        # animation.to_html5_video()
        if (autoplay):
            if (method == 'matplotlib'):
                plt.show()
            elif (method == 'plotly'):
                pt.offline.plot(ani, filename=file_name+'.html')
        
        
    def combination_queue_animation_matplotlib(self, fig, x, y, mode, frames):    
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
        for frame_index in range(0, int(frames)):
            index = int(frame_index*self.time_horizon/frames)
            color_set[:, frame_index] = [ (self.passenger_queuelen[node][mode][index] - self.vehicle_queuelen[node][mode][index]) 
                for node in self.graph.get_graph_dic() ]

        # add another axes at the top left corner of the figure
        axtext = fig.add_axes([0.0,0.95,0.1,0.05])
        # turn the axis labels/spines/ticks off
        axtext.axis('off')
        # place the text to the other axes
        time = axtext.text(0.5,0.5, 'time step={}'.format(0), ha='left', va='top')

        def update(frame):
            scat.set_sizes( scale )
            scat.set_array( color_set[:, frame%frames] )
            time.set_text('time step={}'.format(int(frame*self.time_horizon/frames)))
            return scat,time,

        print('Generate passenger and {} queue ......'.format(mode), end='')
        # Construct the animation, using the update function as the animation director.
        ani = animation.FuncAnimation(fig=fig, func=update, interval=300, frames=frames, repeat=True)
        return ani

    def combination_queue_animation_plotly(self, fig, x, y, mode, frames):
        color = [ (self.passenger_queuelen[node][mode][0] - self.vehicle_queuelen[node][mode][0]) 
            for node in self.graph.get_graph_dic() ]
        scale = [ 300 if (',' in self.graph.get_graph_dic()[node]['mode']) else 100 for node in self.graph.get_graph_dic() ]

        result = np.array([ (self.passenger_queuelen[node][mode] - self.vehicle_queuelen[node][mode])
            for node in self.vehicle_queuelen ])

        color_set = np.zeros(shape=(len(self.graph.get_graph_dic()), frames))
        scale_set = np.zeros(shape=(len(self.graph.get_graph_dic()), frames))
        for frame_index in range(0, int(frames)):
            index = int(frame_index*self.time_horizon/frames)
            color_set[:, frame_index] = [ (self.passenger_queuelen[node][mode][index] - self.vehicle_queuelen[node][mode][index]) 
                for node in self.graph.get_graph_dic() ]
            scale_set[:, frame_index] = [ np.abs(self.passenger_queuelen[node][mode][index] - self.vehicle_queuelen[node][mode][index]) +100
                for node in self.graph.get_graph_dic() ]            

        fig = self.plotly_sactter_animation_data(frames=frames, x=x, y=y, color=color_set, scale=scale_set, result=result)
        
        return fig


    def combination_queue_animation_plotly_3d(self, fig, x, y, mode, frames):
                    

        def bar_data(position3d, size=(1,1,1)):
            # position3d - 3-list or array of shape (3,) that represents the point of coords (x, y, 0), where a bar is placed
            # size = a 3-tuple whose elements are used to scale a unit cube to get a paralelipipedic bar
            # returns - an array of shape(8,3) representing the 8 vertices of  a bar at position3d
            
            bar = np.array([[0, 0, 0],[1, 0, 0],[1, 1, 0],[0, 1, 0],[0, 0, 1],[1, 0, 1],[1, 1, 1],[0, 1, 1]], dtype=float) # the vertices of the unit cube
        
            bar *= np.asarray(size)# scale the cube to get the vertices of a parallelipipedic bar
            bar += np.asarray(position3d) #translate each  bar on the directio OP, with P=position3d
            return bar

        def triangulate_bar_faces(positions, sizes=None):
            # positions - array of shape (N, 3) that contains all positions in the plane z=0, where a histogram bar is placed 
            # sizes -  array of shape (N,3); each row represents the sizes to scale a unit cube to get a bar
            # returns the array of unique vertices, and the lists i, j, k to be used in instantiating the go.Mesh3d class

            if sizes is None:
                sizes = [(1,1,1)]*len(positions)
            else:
                if isinstance(sizes, (list, np.ndarray)) and len(sizes) != len(positions):
                    raise ValueError('Your positions and sizes lists/arrays do not have the same length')
                    
            all_bars = [bar_data(pos, size)  for pos, size in zip(positions, sizes) if size[2]!=0]
            p, q, r = np.array(all_bars).shape
            # extract unique vertices from the list of all bar vertices
            vertices, ixr = np.unique(np.array(all_bars).reshape(p*q, r), return_inverse=True, axis=0)
            # print(vertices)
            #for each bar, derive the sublists of indices i, j, k assocated to its chosen  triangulation
            I = []
            J = []
            K = []
            for k in range(len(all_bars)):
                I.extend(np.take(ixr, [8*k, 8*k+2,8*k, 8*k+5,8*k, 8*k+7, 8*k+5, 8*k+2, 8*k+3, 8*k+6, 8*k+7, 8*k+5])) 
                J.extend(np.take(ixr, [8*k+1, 8*k+3, 8*k+4, 8*k+1, 8*k+3, 8*k+4, 8*k+1, 8*k+6, 8*k+7, 8*k+2, 8*k+4, 8*k+6])) 
                K.extend(np.take(ixr, [8*k+2, 8*k, 8*k+5, 8*k, 8*k+7, 8*k, 8*k+2, 8*k+5, 8*k+6, 8*k+3, 8*k+5, 8*k+7]))  
            return  vertices, I, J, K  #triangulation vertices and I, J, K for mesh3d

        def get_plotly_mesh3d(x, y, z, bins=[5,5], bargap=0.05, range_extent=0.2):
            # x, y- array-like of shape (n,), defining the x, and y-ccordinates of data set for which we plot a 3d hist
            x_range = [np.min(x)-range_extent, np.max(x)+range_extent]
            y_range = [np.min(y)-range_extent, np.max(y)+range_extent]
            hist, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[x_range, y_range])
            
            hist = np.zeros([bins[0], bins[1]])
            for (x_index, x_value) in enumerate(x):
                xpos = int((x_value-x_range[0])/((x_range[0]-x_range[1])/(bins[0]-1)))
                ypos = int((y[x_index]-y_range[0])/((y_range[0]-y_range[1])/(bins[1]-1)))
                #print(xpos, ypos)
                hist[xpos][ypos] = z[x_index]
            
            xsize = xedges[1]-xedges[0]-bargap
            ysize = yedges[1]-yedges[0]-bargap
            xe, ye= np.meshgrid(xedges[:-1], yedges[:-1])
            ze = np.zeros(xe.shape)

            positions =np.dstack((xe, ye, ze))
            m, n, p = positions.shape
            positions = positions.reshape(m*n, p)
            # print(hist.flatten())
            sizes = np.array([(xsize, ysize, h) for h in hist.flatten()])
            # print('sizes:',sizes)
            vertices, I, J, K  = triangulate_bar_faces(positions, sizes=sizes)
            X, Y, Z = vertices.T
            return X, Y, Z, I, J, K


        z = np.zeros(shape=(len(self.graph.get_graph_dic()), frames))
        data_list = []
        for frame_index in range(0, int(frames)):
            index = int(frame_index*self.time_horizon/frames)
            z[:, frame_index] = np.asarray([ (self.passenger_queuelen[node][mode][index] - self.vehicle_queuelen[node][mode][index]) 
                for node in self.graph.get_graph_dic() ])
            
            X, Y, Z, I, J, K = get_plotly_mesh3d(x, y, z[:, frame_index], bins =[500, 500], bargap=2*1e-5)
            data = go.Mesh3d(x=X, y=Y, z=z[:, frame_index], i=I, j=J, k=K, color="#ba2461", flatshading=True)
            # print(data)
            data_list.append(data)

        fig = self.plotly_3d_animation_data(frames=frames, data=data_list)
        # layout = go.Layout(width=1200, height=900, title_text='3D Bar Chart', title_x =0.5)

        # fig = go.Figure(data=[mesh3d], layout=layout)
        # pt.offline.plot(fig)    
        return fig


    def combination_queue_animation(self, mode, frames, autoplay=False, autosave=False, method='matplotlib'):
        lat = [ self.graph.get_node_location(node)[0] for node in self.graph.get_graph_dic() ]
        lon = [ self.graph.get_node_location(node)[1] for node in self.graph.get_graph_dic() ]

        fig = self.graph.plot_topology_edges(lon, lat, method)

        if (method == 'matplotlib'):
            ani = self.combination_queue_animation_matplotlib(fig=fig, x=lon, y=lat, mode=mode, frames=frames)
        elif (method == 'plotly'):
            ani = self.combination_queue_animation_plotly(fig=fig, x=lon, y=lat, mode=mode, frames=frames)
        elif (method == 'plotly_3d'):
            ani = self.combination_queue_animation_plotly_3d(fig=fig, x=lon, y=lat, mode=mode, frames=frames)

        file_name = 'results/{}_combined_queue'.format(mode)
        try:
            os.remove(file_name+'.mp4')
        except OSError:
            # logging.warning('Delete log file failed'.format(figs_path))
            pass
        
        if (autosave):
            ani.save(file_name+'.mp4', fps=12, dpi=300)

        print('Done')
        # animation.to_html5_video()
        if (autoplay):
            if (method == 'matplotlib'):
                plt.show()
            elif (method == 'plotly' or method == 'plotly_3d'):
                pt.offline.plot(ani, filename=file_name+'.html')


