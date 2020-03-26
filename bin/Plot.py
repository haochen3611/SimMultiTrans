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
        with open(f'{path_name}/passegner_queue.json', 'w') as json_file:
            queue_p = json.load(json_file)

        with open(f'{path_name}/vehicle_queue.json', 'w') as json_file:
            queue_v = json.load(json_file)

        with open(f'{path_name}/wait_time.json', 'w') as json_file:
            waittime = json.load(json_file)

        with open(f'{path_name}/total_distance.json', 'w') as json_file:
            totaldist = json.load(json_file)

        for node in self.graph.graph_top:
            for mode in self.vehicle_attri:
                queue_p[node][mode] = np.array(queue_p[node][mode].tolist())
                queue_v[node][mode] = np.array(queue_v[node][mode].tolist())
                waittime[node][mode] = np.array(waittime[node][mode].tolist())


    def import_queuelength(self, queue_p, queue_v):
        self.queue_p = queue_p
        self.queue_v = queue_v
    
    def import_passenger_waittime(self, waittime):
        self.passenger_waittime = waittime

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
        fig.update_layout(template='plotly_dark')
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
        fig.update_layout(template='plotly_dark')
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
        
    def passenger_queue_animation_matplotlib(self, fig, mode, frames):
        data = np.zeros(shape=(len(self.graph.graph_top), frames))
        for frame in range(0, int(frames)):
            queue[:, frame] = [ self.queue_p[node][mode][ int(frame*self.time_horizon/frames) ] 
                for node in self.graph.graph_top ]

        result = np.array([self.queue_p[node][mode] for node in self.queue_p])

        norm = mpl.colors.Normalize(vmin=0, vmax=result.max())

        scat = plt.scatter(x=self.lon, y=self.lat, c=data[:, 0], s=np.abs(data[:, 0])+100, cmap='Reds', label=data[:, 0],
                             norm=norm, zorder=2, alpha=0.8, edgecolors='none')
        cbar = plt.colorbar()

        # add another axes at the top left corner of the figure
        axtext = fig.add_axes([0.0,0.95,0.1,0.05])
        # turn the axis labels/spines/ticks off
        axtext.axis('off')
        # place the text to the other axes
        time = axtext.text(0.5,0.5, f'time step={0}', ha='left', va='top')

        def update(frame):
            scat.set_sizes( np.abs(data[:, frame%frames])+100 )
            scat.set_array( data[:, frame%frames] )
            time.set_text(f'time step={int(frame*self.time_horizon/frames)}')
            return scat,time,

        print(f'Generate {mode} Passenger queue ......', end='')
        # Construct the animation, using the update function as the animation director.
        ani = animation.FuncAnimation(fig=fig, func=update, interval=50, frames=frames, repeat=True)
        return ani


    def passenger_queue_animation_plotly(self, fig, mode, frames):
        data = np.zeros(shape=(len(self.graph.graph_top), frames))
        for frame_index in range(0, int(frames)):
            data[:, frame_index] = [ self.queue_p[node][mode][ int(frame_index*self.time_horizon/frames) ] 
                for node in self.graph.graph_top ]

        fig = self.plotly_sactter_animation_data(frames=frames, lon=self.lon, lat=self.lat, data=data)
        
        return fig


    def passenger_queue_animation(self, mode, frames, autoplay=False, autosave=False, method='matplotlib'):
        if (method == 'matplotlib'):
            fig = self.graph.plot_topology_edges(self.lon, self.lat, method)
            ani = self.passenger_queue_animation_matplotlib(fig=fig, mode=mode, frames=frames)
        elif (method == 'plotly'):
            fig = None
            ani = self.passenger_queue_animation_plotly(fig=fig, mode=mode, frames=frames)
        
        file_name = 'results/passenger_queue'
        try:
            os.remove(file_name+'.mp4')
        except OSError:
            pass
        
        if (autosave and method == 'matplotlib'):
            ani.save(file_name+'.mp4', fps=12, dpi=300)

        print('Done')
        # animation.to_html5_video()
        if (autoplay):
            if (method == 'matplotlib'):
                plt.show()
            elif (method == 'plotly'):
                pt.offline.plot(ani, filename=file_name+'.html')


    def vehicle_queue_animation_matplotlib(self, fig, mode, frames):
        data = np.zeros(shape=(len(self.graph.graph_top), frames))
        for frame_index in range(0, int(frames)):
            data[:, frame_index] = [ self.queue_v[node][mode][ int(frame_index*self.time_horizon/frames) ] 
                for node in self.graph.graph_top ]

        norm = mpl.colors.Normalize(vmin=data.max(), vmax=data.max())

        scat = plt.scatter(x=self.lon, y=self.lat, c=data, s=np.abs(data)+100, cmap='Blues', 
                    label=data, norm=norm, zorder=2, alpha=0.8, edgecolors='none')
        cbar = plt.colorbar()

        # add another axes at the top left corner of the figure
        axtext = fig.add_axes([0.0,0.95,0.1,0.05])
        # turn the axis labels/spines/ticks off
        axtext.axis('off')
        # place the text to the other axes
        time = axtext.text(0.5,0.5, f'time step={0}', ha='left', va='top')

        def update(frame):
            scat.set_sizes( np.abs(data[:, frame%frames])+100 )
            scat.set_array( data[:, frame%frames] )
            time.set_text(f'time step={int(frame*self.time_horizon/frames)}')
            return scat,time,

        print(f'Generate {mode} queue ......', end='')
        # Construct the animation, using the update function as the animation director.
        ani = animation.FuncAnimation(fig=fig, func=update, interval=50, frames=frames, repeat=True)
        return ani


    def vehicle_queue_animation_plotly(self, fig, mode, frames):
        data = np.zeros(shape=(len(self.graph.graph_top), frames))
        for frame_index in range(0, int(frames)):
            data[:, frame_index] = [ self.queue_v[node][mode][ int(frame_index*self.time_horizon/frames) ] 
                for node in self.graph.graph_top ]        

        fig = self.plotly_sactter_animation_data(frames=frames, lon=self.lon, lat=self.lat, data=data)
        return fig


    def vehicle_queue_animation(self, mode, frames, autoplay=False, autosave=False, method='matplotlib'):
        if (method == 'matplotlib'):
            fig = self.graph.plot_topology_edges(self.lon, self.lat, method)
            ani = self.vehicle_queue_animation_matplotlib(fig=fig, mode=mode, frames=frames)
        elif (method == 'plotly'):
            fig = None
            ani = self.vehicle_queue_animation_plotly(fig=fig, mode=mode, frames=frames)

        file_name = f'results/{mode}_queue'
        try:
            os.remove(file_name+'.mp4')
        except OSError:
            pass

        if (autosave and method == 'matplotlib'):
            ani.save(file_name+'.mp4', fps=12, dpi=300)

        print('Done')
        # animation.to_html5_video()
        if (autoplay):
            if (method == 'matplotlib'):
                plt.show()
            elif (method == 'plotly'):
                pt.offline.plot(ani, filename=file_name+'.html')
        
        
    def combination_queue_animation_matplotlib(self, fig, mode, frames):    
        data = np.zeros(shape=(len(self.graph.graph_top), frames))

        # sum all buses
        bus_list = []
        if (mode == 'bus'):
            for node in self.graph.graph_top:
                modelist =  self.graph.graph_top[node]['mode'].split(',')
                modelist = [ bus for bus in modelist if ( 'BUS' in bus ) ]
                print(modelist)
                bus_list = list(set(bus_list + modelist))
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

        # norm = mpl.colors.Normalize(vmin=result.min(), vcenter=0, vmax=result.max())
        norm = MidpointNormalize(vmin=data.min(), vcenter=0, vmax=data.max())

        scat = plt.scatter(x=self.lon, y=self.lat, c=data[:, 0], s=np.abs(data[:, 0])+100, cmap='coolwarm', 
                    label=data[:, 0], norm=norm, zorder=2, alpha=0.8, edgecolors='none')
        cbar = plt.colorbar()
        

        # add another axes at the top left corner of the figure
        axtext = fig.add_axes([0.0,0.95,0.1,0.05])
        # turn the axis labels/spines/ticks off
        axtext.axis('off')
        # place the text to the other axes
        time = axtext.text(0.5,0.5, f'time step={0}', ha='left', va='top')

        def update(frame):
            scat.set_sizes( np.abs(data[:, frame%frames])+100 )
            scat.set_array( data[:, frame%frames] )
            time.set_text(f'time step={int(frame*self.time_horizon/frames)}')
            return scat,time,

        print(f'Generate passenger and {mode} queue ......', end='')
        # Construct the animation, using the update function as the animation director.
        ani = animation.FuncAnimation(fig=fig, func=update, interval=300, frames=frames, repeat=True)
        return ani


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


    def combination_queue_animation(self, mode, frames, autoplay=False, autosave=False, method='matplotlib'):
        self.lat = [ self.graph.graph_top[node]['lat'] for node in self.graph.graph_top ]
        self.lon = [ self.graph.graph_top[node]['lon'] for node in self.graph.graph_top ]

        if (method == 'matplotlib'):
            fig = self.graph.plot_topology_edges(self.lon, self.lat, method)
            ani = self.combination_queue_animation_matplotlib(fig=fig, mode=mode, frames=frames)
        elif (method == 'plotly'):
            fig = None
            ani = self.combination_queue_animation_plotly(fig=fig, mode=mode, frames=frames)

        file_name = f'results/{mode}_combined_queue'
        try:
            os.remove(file_name+'.mp4')
        except OSError:
            pass
        
        if (autosave and method == 'matplotlib'):
            ani.save(file_name+'.mp4', fps=12, dpi=300)

        print('Done')
        # animation.to_html5_video()
        if (autoplay):
            if (method == 'matplotlib'):
                plt.show()
            elif (method == 'plotly'):
                pt.offline.plot(ani, filename=file_name+'.html')


    def plot_passenger_queuelen_time(self, mode, method='plotly'):
        fig = go.Figure()
        x = np.arange(start=0, stop=self.time_horizon, step=1)
        # x = [(timedelta(seconds=t) + self.start_time).strftime('%H:%M:%S') for t in range(self.time_horizon)]
        # x = range(self.time_horizon)
        # y = np.zeros(len(x))
        
        # cr = (244,67,54)
        # cb = (33,150,243)
        # crange = len(self.queue_p.keys())
        # print(crange)
        for index, node in enumerate(self.queue_p):
            # r = abs(cr[0]-cb[0])*index/crange + min(cr[0],cb[0])
            # g = abs(cr[1]-cb[2])*index/crange + min(cr[1],cb[1])
            # b = abs(cr[2]-cb[2])*index/crange + min(cr[2],cb[2])
            # color = 'rgb({},{},{})'.format(int(r), int(g), int(b))
            # y = y + self.queue_p[node][mode]
            y = self.queue_p[node][mode] - self.queue_v[node][mode]
            # fig.add_trace(go.Scatter(x=x, y=y, name=node, line = {'color': color} ))
            fig.add_trace(go.Scatter(x=x, y=y, name=node))


        fig.update_xaxes(
            ticktext=[(timedelta(seconds=t) + self.start_time).strftime('%H:%M:%S') for t in range(0, self.time_horizon, int(self.time_horizon/10))],
            tickvals=[t for t in range(0, self.time_horizon, int(self.time_horizon/10))],
        )
        
        fig.update_layout(template='plotly_dark')
        file_name = f'results/{mode}_queue_time'
        pt.offline.plot(fig, filename=file_name+'.html')
            
    def plot_passenger_waittime(self, mode, method='plotly'):
        x = self.graph.get_allnodes()
        # print(self.passenger_waittime)
        y = [ self.passenger_waittime[node][mode][-1] for node in self.graph.get_allnodes() ]
        fig = go.Figure(data=[go.Bar(x=x, y=y)])
        fig.update_xaxes(type='category')
        
        fig.update_layout(template='plotly_dark')
        file_name = f'results/{mode}_waittime'
        pt.offline.plot(fig, filename=file_name+'.html')

    def plot_topology(self, method='matplotlib'):
        if (method == 'matplotlib'):
            fig, ax = plt.subplots()
            fig, ax = self.plot_topology_edges(self.lon, self.lat, method)

            # color = np.random.randint(1, 100, size=len(self.get_allnodes()))
            color = [ 'steelblue' if (',' in self.graph_top[node]['mode']) else 'skyblue' for node in self.graph_top ]
            scale = [ 300 if (',' in self.graph_top[node]['mode']) else 100 for node in self.graph_top ]

            ax.scatter(self.lon, self.lat, c=color, s=scale, label=color, alpha=0.8, edgecolors='none', zorder=2)

            # ax.legend()
            plt.grid(False)
            # plt.legend(loc='lower right', framealpha=1)
            plt.xlabel('lat1itude')
            plt.ylabel('Longitude')
            plt.title('City Topology')

            plt.savefig('City_Topology.pdf', dpi=600)
            print(self.graph_top)
            return fig, ax

        elif (method == 'plotly'):
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

            file_name = 'results/Topology'
            fig.update_layout(template='plotly_dark')
            pt.offline.plot(fig, filename=file_name+'.html')

    
    def plot_topology_edges(self, x, y, method='matplotlib'):
        if (method == 'matplotlib'):
            plt.style.use('dark_background')
            fig, ax = plt.subplots()

            alledges = self.graph.get_all_edges()
            # print(alledges)
            loc = np.zeros(shape=(2,2))

            for odlist in alledges:
                for odpair in odlist:
                    loc[:,0] = np.array( [self.graph.graph_top[odpair[0]]['lat'], self.graph.graph_top[odpair[0]]['lon']])
                    loc[:,1] = np.array( [self.graph.graph_top[odpair[1]]['lat'], self.graph.graph_top[odpair[1]]['lon']])
                    ax.plot(loc[0,:], loc[1,:], c='grey', alpha=0.2, ls='--', lw=2, zorder=1)
            return fig
        elif (method == 'plotly'):
            return fig
        else:
            return None



class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))