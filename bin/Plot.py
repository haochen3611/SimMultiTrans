#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import plotly as pt
import plotly.express as px
import plotly.graph_objects as go


import os
import logging
import json
from time import time
from datetime import datetime, timedelta

class Plot(object):
    def __init__(self, graph, time_horizon, start_time):
        self.graph = graph
        self.time_horizon = time_horizon
        self.start_time = start_time
        self.lat = np.asarray([ self.graph.graph_top[node]['lat'] for node in self.graph.graph_top ])
        self.lon = np.asarray([ self.graph.graph_top[node]['lon'] for node in self.graph.graph_top ])

        self.relativesize = 120
        self.basicsize = 6

        try:
            self.mapbox_access_token = open("conf/.mapbox_token").read()
            self.map_style = open("conf/.mapbox_style").read()
        except OSError:
            print('Map Key Error!')
            pass
    
    def import_results(self, path_name):
        self.graph.import_graph('results/city_topology.json')
        
        with open('results/simulation_info.json') as json_file:
            simulation_info = json.load(json_file)
        
        self.time_horizon = simulation_info['Time_horizon']
        self.start_time = datetime.strptime(simulation_info['Start_time'], "%H:%M:%S")
        self.vehicle_attri = simulation_info['Vehicle'] 
        self.reb_method = simulation_info['Rebalancing_method'] 
        self.routing_method = simulation_info['Routing_method'] 
        self.duration = simulation_info['Duration'] 


        self.lat = np.asarray([ self.graph.graph_top[node]['lat'] for node in self.graph.graph_top ])
        self.lon = np.asarray([ self.graph.graph_top[node]['lon'] for node in self.graph.graph_top ])

        with open(f'{path_name}/passenger_queue.json') as json_file:
            queue_p = json.load(json_file)

        with open(f'{path_name}/passenger_queue.json') as json_file:
            queue_p = json.load(json_file)

        with open(f'{path_name}/vehicle_queue.json') as json_file:
            queue_v = json.load(json_file)

        with open(f'{path_name}/wait_time.json') as json_file:
            waittime = json.load(json_file)

        with open(f'{path_name}/metrics.json') as json_file:
            metrics = json.load(json_file)

        self.total_trip= metrics['total_trip']
        self.total_tripdist = metrics['total_tripdist']
        self.total_triptime = metrics['total_triptime']
        self.total_arrival = metrics['total_arrival']
        self.sum_totalarrival = metrics['total_num_arrival']


        for node in self.graph.graph_top:
            for mode in self.vehicle_attri:
                queue_p[node][mode] = np.array(queue_p[node][mode])
                queue_v[node][mode] = np.array(queue_v[node][mode])
                waittime[node][mode] = np.array(waittime[node][mode])

        self.queue_p = queue_p
        self.queue_v = queue_v
        self.waittime_p = waittime


    def set_plot_theme(self, theme):
        if (theme in ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]):
            pt.io.templates.default = theme
        else:
            pt.io.templates.default = 'simple_white'

    def plot_passenger_queuelen(self, mode, time):
        time_step = (datetime.strptime(time, '%H:%M:%S')-self.start_time).seconds

        if (time_step < 0 or time_step > self.time_horizon):
            return

        fig_dict = {'data': [], 'layout': {}}

        # fill in most of layout
        fig_dict['layout']['xaxis'] = {'title': 'Latitude'}
        fig_dict['layout']['yaxis'] = {'title': 'Longitude'}
        fig_dict['layout']['hovermode'] = 'closest'
        fig_dict['layout']['mapbox'] = {
            'accesstoken': self.mapbox_access_token,
            'bearing': 0,
            'center': go.layout.mapbox.Center(
                lat = np.mean(self.lat),
                lon = np.mean(self.lon)
            ),
            'pitch': 60,
            'zoom': 11,
            'style': self.map_style
        }

        data = np.array([ self.queue_v[node][mode][ time_step ] for node in self.graph.graph_top ])
        cmin = np.min(data.min(), 0)
        cmax = np.max(data.max(), 0)
        cmax = cmin + 1 if (cmax - cmin == 0) else cmax
        colorsacle = [ [0, '#33691E'], [np.abs(cmin)/(cmax - cmin), '#FAFAFA'], [1, '#FF6F00'] ]
        sizescale = self.relativesize/np.max( [cmax, np.abs(cmin)] )
        text_str = [f'{self.graph.get_allnodes()[index]}: {data[index]}' for index in range(len(data))]
        data_dict = { 
            'type':'scattermapbox', 
            'lon': self.lon, 'lat': self.lat, 
            'mode': 'markers', 
            'name': 'Queue', 
            'text': text_str,
            'marker': { 
                'size': data*sizescale + self.basicsize, 
                # 'size': np.log(data+self.relativesize), 
                'color': data, 'colorscale': colorsacle,
                'cmin': data.min(), 'cmax': data.max(), 'colorbar': dict(title='Queue')  
            }
        }
        fig_dict['data'].append(data_dict)
        fig = go.Figure(fig_dict)

        file_name = f'results/Passenger_{mode}_queue_at_{time}'
        # fig.update_layout(template='plotly_dark')
        pt.offline.plot(fig, filename=file_name+'.html')


    def plotly_sactter_animation_data(self, frames, lon, lat, data):
        fig_dict = {'data': [], 'layout': {}, 'frames': []}

        # fill in most of layout
        fig_dict['layout']['xaxis'] = {'title': 'Latitude'}
        fig_dict['layout']['yaxis'] = {'title': 'Longitude'}
        fig_dict['layout']['hovermode'] = 'closest'
        fig_dict['layout']['mapbox'] = {
            'accesstoken': self.mapbox_access_token,
            'bearing': 0,
            'center': go.layout.mapbox.Center(
                lat=np.mean(self.lat),
                lon=np.mean(self.lon)
            ),
            'pitch': 60,
            'zoom': 11,
            'style': self.map_style
        }
        fig_dict['layout']['sliders'] = {
            'args': [ 'transition', { 'duration': 400, 'easing': 'cubic-in-out' } ],
            'initialValue': '0', 'plotlycommand': 'animate', 'values': range(frames), 'visible': True
        }
        fig_dict['layout']['updatemenus'] = [ {
                'buttons': [ 
                    { 'args': [None, {'frame': {'duration': 500, 'redraw': True},
                      'fromcurrent': True, 'transition': {'duration': 300, 'easing': 'quadratic-in-out'}}],
                      'label': 'Play', 'method': 'animate' },
                    { 'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}],
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
        cmin = np.min([data.min(), 0])
        cmax = np.max([data.max(), 0])
        cmax = cmin + 1 if (cmax - cmin == 0) else cmax
        colorsacle = [ [0, '#33691E'], [np.abs(cmin)/(cmax - cmin), '#FAFAFA'], [1, '#FF6F00'] ]
        sizescale = self.relativesize/np.max( [cmax, np.abs(cmin)] )
        text_str = [f'{self.graph.get_allnodes()[index]}: {data[index, 0]}' for index in range(len(data))]
        data_dict = { 
            'type':'scattermapbox', 
            'lon': lon, 'lat': lat, 
            'mode': 'markers', 
            'name': 'Queue', 
            'text': text_str,
            'marker': { # 'size': np.abs(data[:, 0])+self.relativesize, 
                        'size': np.abs(data[:, 0])*sizescale + self.basicsize,
                        'color': data[:, 0], 'colorscale': colorsacle,
                        'cmin': cmin, 'cmax': cmax, 'colorbar': dict(title='Queue')  }
        }
        fig_dict['data'].append(data_dict)

        # make frames
        for frame_index in range(frames):
            frame = {'data': [], 'name': str(frame_index)}
            text_str = [f'{self.graph.get_allnodes()[index]}: {data[index, frame_index]}' for index in range(len(data))]
            data_dict = { 
                'type':'scattermapbox', 
                'lon': lon, 'lat': lat, 
                'mode': 'markers', 
                'name': 'Queue', 
                'text': text_str,
                'marker': { 
                    'size': np.abs(data[:, frame_index])*sizescale + self.basicsize,
                    # 'size': 5*np.log(np.abs(data[:, frame_index])+self.relativesize), 
                    'color': data[:, frame_index], 'colorscale': colorsacle,
                    'cmin': cmin, 'cmax': cmax, 'colorbar': dict(title='Queue')  }
            }

            frame['data'].append(data_dict)

            fig_dict['frames'].append(frame)
            frame_time = timedelta(seconds=frame_index*self.time_horizon/frames) + self.start_time
            slider_step = {'args': [ 
                [frame_index], {'frame': {'duration': 300, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 300}} ],
                'label': frame_time.strftime('%H:%M'), 'method': 'animate'}
            sliders_dict['steps'].append(slider_step)

        fig_dict['layout']['sliders'] = [sliders_dict]

        fig = go.Figure(fig_dict)
        # fig.update_layout(template='plotly_dark')
        return fig

    '''
    def plotly_3d_animation_data(self, frames, data):

        fig_dict = {'data': [], 'layout': {}, 'frames': []}

        # fill in most of layout
        fig_dict['layout']['xaxis'] = {'title': 'Latitude'}
        fig_dict['layout']['yaxis'] = {'title': 'Longitude'}
        # fig_dict['layout']['hovermode'] = 'closest'
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
        
        data_dict = { 'x': x, 'y': y, 'mode': 'markers', 'name': 'Queue', 'text': self.graph.get_allnodes(),
            'marker': { 'sizemode': 'area', 'size': scale[:, 0], 'sizeref': 2.*max(scale[:, 0])/(40.**2),
                        'color': color[:, 0], 'colorscale': colorsacle,
                        'cmin': result.min(), 'cmax': result.max(), 'colorbar': dict(title='Queue')  }
                        }

        # data_dict = data[0]
        # fig_dict['data'].append(data[0])
        f = go.Mesh3d(x=data[0]['X'], y=data[0]['Y'], z=data[0]['Z'], 
                i=data[0]['I'], j=data[0]['J'], k=data[0]['K'], color="#ba2461", flatshading=True)
        fig_dict['data'].append(f)

        # make frames
        for frame_index in range(frames):
            frame = {'data': [], 'name': str(frame_index)}
            f = go.Mesh3d(x=data[frame_index]['X'], y=data[frame_index]['Y'], z=data[frame_index]['Z'], 
                i=data[frame_index]['I'], j=data[frame_index]['J'], k=data[frame_index]['K'], color="#ba2461", flatshading=True)
            frame['data'].append( f )

            fig_dict['frames'].append(frame)
            frame_time = timedelta(seconds=frame_index*self.time_horizon/frames) + self.start_time

            slider_step = {'args': [ 
                [frame_index], {'frame': {'duration': 300, 'redraw': False}, 'mode': 'immediate', 'transition': {'duration': 300}} ],
                'label': frame_time.strftime('%H:%M'), 'method': 'animate'}
            sliders_dict['steps'].append(slider_step)
            # print(sliders_dict['steps'][frame_index])

        fig_dict['layout']['sliders'] = [sliders_dict]
        
        fig = go.Figure(fig_dict)
        fig.update_layout(template='plotly_dark')
        return fig
    '''

    def passenger_queue_animation_plotly(self, fig, mode, frames):
        data = np.zeros(shape=(len(self.graph.graph_top), frames))
        for frame_index in range(0, int(frames)):
            data[:, frame_index] = [ self.queue_p[node][mode][ int(frame_index*self.time_horizon/frames) ] 
                for node in self.graph.graph_top ]

        fig = self.plotly_sactter_animation_data(frames=frames, lon=self.lon, lat=self.lat, data=data)
        
        return fig


    def passenger_queue_animation(self, mode, frames, autoplay=False, autosave=False):
        print(f'Plot queue length of passengers who take {mode} ...', end='')
        fig = None
        ani = self.passenger_queue_animation_plotly(fig=fig, mode=mode, frames=frames)
            
        pt.offline.plot(ani, filename='results/passenger_queue.html')
        print('Done')
            


    def vehicle_queue_animation_plotly(self, fig, mode, frames):
        data = np.zeros(shape=(len(self.graph.graph_top), frames))
        for frame_index in range(0, int(frames)):
            data[:, frame_index] = [ self.queue_v[node][mode][ int(frame_index*self.time_horizon/frames) ] 
                for node in self.graph.graph_top ]        

        fig = self.plotly_sactter_animation_data(frames=frames, lon=self.lon, lat=self.lat, data=data)
        return fig


    def vehicle_queue_animation(self, mode, frames, autoplay=False, autosave=False):
        print(f'Plot queue length of {mode} ...', end='')
        fig = None
        ani = self.vehicle_queue_animation_plotly(fig=fig, mode=mode, frames=frames)
            
        pt.offline.plot(ani, filename=f'results/{mode}_queue.html')
        print('Done')
                


    def combination_queue_animation_plotly(self, fig, mode, frames):
        data = np.zeros(shape=(len(self.graph.graph_top), frames))
        # sum all buses
        bus_list = []
        if (mode == 'bus'):
            for node in self.graph.graph_top:
                modelist =  self.graph.graph_top[node]['mode'].split(',')
                modelist = [ bus for bus in modelist if ( 'BUS' in bus ) ]
                # print(modelist)
                bus_list = list(set(bus_list + modelist))
            # print(bus_list)
            for bus in bus_list:
                for frame_index in range(0, int(frames)):
                    index = int(frame_index*self.time_horizon/frames)
                    data[:, frame_index] = data[:, frame_index] + [ (self.queue_p[node][bus][index] - self.queue_v[node][bus][index]) 
                            for node in self.graph.graph_top ]
        else:
            for frame_index in range(0, int(frames)):
                index = int(frame_index*self.time_horizon/frames)
                data[:, frame_index] = [ (self.queue_p[node][mode][index] - self.queue_v[node][mode][index]) 
                    for node in self.graph.graph_top ]

        fig = self.plotly_sactter_animation_data(frames=frames, lon=self.lon, lat=self.lat, data=data)
        return fig


    def combination_queue_animation(self, mode, frames, autoplay=False, autosave=False):
        print(f'Plot combined queue length of passengers and {mode} ...', end='')

        self.lat = [ self.graph.graph_top[node]['lat'] for node in self.graph.graph_top ]
        self.lon = [ self.graph.graph_top[node]['lon'] for node in self.graph.graph_top ]

        fig = None
        ani = self.combination_queue_animation_plotly(fig=fig, mode=mode, frames=frames)

        pt.offline.plot(ani, filename=f'results/{mode}_combined_queue.html')

        print('Done')
                


    def plot_passenger_queuelen_time(self, mode):
        fig_dict = {'data': [], 'layout': {}}
        fig_dict['layout']['xaxis'] = {'title': 'Time'}
        fig_dict['layout']['xaxis']['ticktext'] = [
            (timedelta(seconds=t) + self.start_time).strftime('%H:%M:%S') for t in range(0, self.time_horizon, int(self.time_horizon/10))
        ]
        fig_dict['layout']['xaxis']['tickvals'] = [ t for t in range(0, self.time_horizon, int(self.time_horizon/10)) ]
        fig_dict['layout']['yaxis'] = {'title': f'Imbalance: # passengers - # {mode}'}
        fig_dict['layout']['hovermode'] = 'closest'
        fig_dict['layout']['title'] = 'Changing of Imbalance'
        
        # fig = go.Figure()
        x = np.arange(start=0, stop=self.time_horizon, step=1)

        for index, node in enumerate(self.queue_p):
            # does not plot the node only with walk
            if (self.graph.graph_top[node]['mode'] != 'walk'):
                y = self.queue_p[node][mode] - self.queue_v[node][mode]
                data_dict = { 
                    'type':'scatter', 'x': x, 'y': y, 'name': node
                }
                fig_dict['data'].append(data_dict)

        fig = go.Figure(fig_dict)
        
        # fig.update_layout(template='plotly_dark')
        file_name = f'results/{mode}_queue_time'
        pt.offline.plot(fig, filename=file_name+'.html')
            
    def plot_passenger_waittime(self, mode):
        fig_dict = {'data': [], 'layout': {}}
        fig_dict['layout']['xaxis'] = {'title': 'Region ID'}
        fig_dict['layout']['xaxis']['type'] = 'category'
        fig_dict['layout']['yaxis'] = {'title': 'Waiting Time (s)'}
        fig_dict['layout']['hovermode'] = 'closest'
        fig_dict['layout']['title'] = 'Average Waiting Time'
        
        x = [ node for node in self.graph.get_allnodes() if (self.graph.graph_top[node]['mode'] != 'walk') ]
        # print(self.passenger_waittime)
        y = [ self.waittime_p[node][mode] for node in x ]
        data_dict = { 
            'type':'bar', 'x': x, 'y': y, 'marker_color': 'lightsalmon', 'textposition': 'auto'
        }
        fig_dict['data'].append(data_dict)
        fig = go.Figure(fig_dict)        

        file_name = f'results/{mode}_waittime'
        pt.offline.plot(fig, filename=file_name+'.html')

    def plot_metrics(self, mode):
        # fig = pt.subplots.make_subplots(rows=2, cols=2)
        fig_dict = {'data': [], 'layout': {}}

        fig_dict['layout']['hovermode'] = 'closest'
        fig_dict['layout']['title'] = 'Operational Metrics'
        fig_dict['layout']['showlegend'] = False
        
        # average waiting time
        x = [ node for node in self.graph.get_allnodes() if (self.graph.graph_top[node]['mode'] != 'walk') ]
        # print(self.passenger_waittime)
        y = [ self.waittime_p[node][mode]/60.0 for node in x ]
        data_dict = { 
            'type':'bar', 'x': x, 'y': y, 'name': 'Waiting Time', 'offsetgroup': '0',
            'marker_color': 'indianred', 'xaxis': 'x', 'yaxis': 'y'
        }
        fig_dict['data'].append(data_dict)
        fig_dict['layout']['xaxis'] = {'title': 'Region ID', 'type': 'category', 'domain': [0, 1]}
        fig_dict['layout']['yaxis'] = {'title': 'Averaged Waiting Time (min)', 'titlefont' : {'color': 'indianred'},
            'domain': [0, 0.45], 'anchor': 'x', 'overlaying': 'y2'}

        # passenger throughtput
        y = [ (self.total_trip['total'][node][mode]-self.total_trip['reb'][node][mode])/float(self.time_horizon)*3600
             for node in x ]
        data_dict = { 
            'type':'bar', 'x': x, 'y': y, 'name': 'Throughput', 'offsetgroup': '1',
            'marker_color': 'lightsalmon', 'xaxis': 'x', 'yaxis': 'y2'
        }
        fig_dict['data'].append(data_dict)

        fig_dict['layout']['yaxis2'] = {'title': 'Throughput (#/min)', 'titlefont' : {'color': 'lightsalmon'}, 
            'domain': [0, 0.45], 'anchor': 'x', 'side': 'right'}
        
        # trip time/dist
        # x = ['Trip Time', 'Trip Distance']
        sum_trip = {'total': 0, 'reb': 0}
        for node in x:
            sum_trip['total'] += self.total_trip['total'][node][mode]
            sum_trip['reb'] += self.total_trip['reb'][node][mode]
        
        x = ['Riding Distance', 'Rebalancing Distance']
        y = np.array([
            (self.total_tripdist['total'][mode]-self.total_tripdist['reb'][mode])/float(sum_trip['total']-sum_trip['reb']), 
            self.total_tripdist['reb'][mode]/float(sum_trip['reb'])
        ])
        color = ['#81D4FA', '#0288D1']
        data_dict = { 
            'type':'pie', 'labels': x, 'values': y, 'textposition': 'inside', 'textinfo': 'percent+label', 'name': 'Total Trip Distance',
            'marker_colors': color, 'domain':{'x': [0.55, 0.75], 'y': [0.55, 1]},
        }
        fig_dict['data'].append(data_dict)

        x = ['Riding Time', 'Rebalancing Time']
        y = np.array([
            (self.total_triptime['total'][mode]-self.total_triptime['reb'][mode])/float(sum_trip['total']-sum_trip['reb']), 
            self.total_triptime['reb'][mode]/float(sum_trip['reb'])
        ])
        color = ['#81D4FA', '#0288D1']
        data_dict = { 
            'type':'pie', 'labels': x, 'values': y, 'textposition': 'inside', 'textinfo': 'percent+label', 'name': 'Total Trip Distance',
            'marker_colors': color, 'domain':{'x': [0.8, 1], 'y': [0.55, 1]},
        }
        fig_dict['data'].append(data_dict)

        # general metrics
        # x = ['Total Trips', 'Total Miles Traveled', 'Total Hours Traveled']
        # y = [sum_trip['total'], self.total_tripdist['total'][mode], self.total_triptime['total'][mode]]
        # color = ['#66BB6A', '#9CCC65', '#D4E157']
        # total trips
        x = ['Total Trips']
        y = [sum_trip['total']]
        data_dict = { 
            'type':'bar', 'x': x, 'y': y, 'name': '', 'text': y, 'textposition': 'outside',
            'marker_color': '#66BB6A', 'xaxis': 'x2', 'yaxis': 'y3'
        }
        fig_dict['data'].append(data_dict)
        fig_dict['layout']['xaxis2'] = {'title': 'General Metrics', 'type': 'category', 'domain': [0, 0.45], 'anchor': 'y3'}
        fig_dict['layout']['yaxis3'] = {'title': '', 'showgrid': False, 'ticks': '', 'showticklabels': False,
            'domain': [0.55, 1], 'anchor': 'x2', 'overlaying': 'y5', 'range': [0, y[0]*5.4]}

        x = ['Total Miles Traveled']
        y = [self.total_tripdist['total'][mode]/1609.34]
        data_dict = { 
            'type':'bar', 'x': x, 'y': np.around(y,decimals=2), 'name': '', 'text': y, 'textposition': 'outside',
            'marker_color': '#9CCC65', 'xaxis': 'x2', 'yaxis': 'y4'
        }
        fig_dict['data'].append(data_dict)
        fig_dict['layout']['yaxis4'] = {'title': '', 'showgrid': False, 'ticks': '', 'showticklabels': False,
            'domain': [0.55, 1], 'anchor': 'x2', 'overlaying': 'y5', 'range': [0, y[0]*1.4]}

        x = ['Total Hours Traveled']
        y = [self.total_tripdist['total'][mode]/3600.0]
        data_dict = { 
            'type':'bar', 'x': x, 'y': np.around(y,decimals=2), 'name': '', 'text': y, 'textposition': 'outside',
            'marker_color': '#D4E157', 'xaxis': 'x2', 'yaxis': 'y5'
        }
        fig_dict['data'].append(data_dict)
        fig_dict['layout']['yaxis5'] = {'title': '', 'showgrid': False, 'ticks': '', 'showticklabels': False,
            'domain': [0.55, 1], 'anchor': 'x2', 'range': [0, y[0]*2.7]}

        fig = go.Figure(fig_dict)
        
        # fig.update_layout(template='plotly_dark')
        file_name = f'results/{mode}_waittime'
        pt.offline.plot(fig, filename=file_name+'.html')



    def plot_topology(self):
        fig_dict = {'data': [], 'layout': {}}

        # fill in most of layout
        fig_dict['layout']['xaxis'] = {'title': 'Latitude'}
        fig_dict['layout']['yaxis'] = {'title': 'Longitude'}
        fig_dict['layout']['hovermode'] = 'closest'
        fig_dict['layout']['mapbox'] = {
            'accesstoken': self.mapbox_access_token,
            'bearing': 0,
            'center': go.layout.mapbox.Center(
                lat = np.mean(self.lat),
                lon = np.mean(self.lon)
            ),
            'pitch': 60,
            'zoom': 11,
            'style': self.map_style
        }

        size = np.zeros([len(self.lon), 1])+self.relativesize*2
        color = ['#FAFAFA' for node in self.lon]

        text_str = [f'{self.graph.get_allnodes()[index]}' for index in range(len(self.lon))]
        data_dict = { 
            'type':'scattermapbox', 
            'lon': self.lon, 'lat': self.lat, 
            'mode': 'markers', 
            'name': 'Queue', 
            'text': text_str,
            'marker': { 'size': size, 'color': color }
        }
        fig_dict['data'].append(data_dict)
        fig = go.Figure(fig_dict)

        # fig.update_layout(template='plotly_dark')
        pt.offline.plot(fig, filename='results/Topology.html')
            


class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))