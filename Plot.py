#!/usr/bin/env python
# -*- coding: utf-8 -*-
from Converter import MidpointNormalize

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
from time import time
from datetime import datetime, timedelta

class Plot(object):
    def __init__(self, graph, time_horizon, start_time):
        self.graph = graph
        self.time_horizon = time_horizon
        self.start_time = start_time
        self.lat = np.asarray([ self.graph.get_node_location(node)[0] for node in self.graph.get_graph_dic() ])
        self.lon = np.asarray([ self.graph.get_node_location(node)[1] for node in self.graph.get_graph_dic() ])

        self.relativesize = 50
        self.mapbox_access_token = 'pk.eyJ1IjoibW9tb2R1cGkiLCJhIjoiY2s3NzJ5eW12MDNpeTNmbGsyeGt0OXJyOCJ9.ZkQ_HeNeybjIcrLiGpUEtg'

    def import_result(self, queue_p, queue_v):
        self.queue_p = queue_p
        self.queue_v = queue_v

    def plot_passenger_queuelen(self, time):
        fig, ax = self.graph.plot_topology_edges(x=self.lon, y=self.lat)
        # color = np.random.randint(1, 100, size=len(self.get_allnodes()))
        color = [ self.queue_p[node][time] for node in self.graph.get_graph_dic() ]
        scale = [ 300 if (',' in self.graph.get_graph_dic()[node]['mode']) else 100 for node in self.graph.get_graph_dic() ]

        norm = mpl.colors.Normalize(vmin=0, vmax=self.time_horizon)
        
        plt.scatter(x=self.lon, y=self.lat, c=color, s=scale, cmap='Reds', label=color, norm=norm, zorder=2, alpha=0.8, edgecolors='none')
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
        fig_dict['layout']['mapbox'] = {
            'accesstoken': self.mapbox_access_token,
            'bearing': 0,
            'center': go.layout.mapbox.Center(
                lat=np.mean(self.lat),
                lon=np.mean(self.lon)
            ),
            'pitch': 60,
            'zoom': 11,
            'style': 'mapbox://styles/momodupi/ck7754i6n0m9c1io3th9xtd4h'
        }
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
        if (result.max() >= 0):
            zp = np.abs(result.min())/(result.max() - result.min())
        # print(scale[:, 0])
            colorsacle = [ [0, '#33691E'], [zp, '#FAFAFA'], [1, '#FF6F00'] ]
        else: 
            colorsacle = [ [0, '#FAFAFA'], [1, '#FF6F00'] ]
        
        data_dict = { 'type':'scattermapbox', 'lon': x, 'lat': y, 'mode': 'markers', 'name': 'Queue', 'text': self.graph.get_allnodes(),
            'marker': { 'size': scale[:, 0], 'color': color[:, 0], 'colorscale': colorsacle,
                        'cmin': result.min(), 'cmax': result.max(), 'colorbar': dict(title='Queue')  }
        }
        fig_dict['data'].append(data_dict)

        # make frames
        for frame_index in range(frames):
            frame = {'data': [], 'name': str(frame_index)}
            
            data_dict = { 'type':'scattermapbox', 'lon': x, 'lat': y, 'mode': 'markers', 'name': 'Queue', 'text': self.graph.get_allnodes(),
            'marker': { 'size': scale[:, frame_index], 'color': color[:, frame_index], 'colorscale': colorsacle,
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
        color = [ self.queue_p[node][mode][0] for node in self.graph.get_graph_dic() ]
        scale = [ 300 if (',' in self.graph.get_graph_dic()[node]['mode']) else 100 for node in self.graph.get_graph_dic() ]

        result = np.array([self.queue_p[node][mode] for node in self.queue_p])

        norm = mpl.colors.Normalize(vmin=0, vmax=result.max())

        scat = plt.scatter(x=self.lon, y=self.lat, c=color, s=scale, cmap='Reds', label=color, norm=norm, zorder=2, alpha=0.8, edgecolors='none')
        cbar = plt.colorbar()

        color_set = np.zeros(shape=(len(self.graph.get_graph_dic()), frames))
        scale_set = np.zeros(shape=(len(self.graph.get_graph_dic()), frames))
        for frame in range(0, int(frames)):
            color_set[:, frame] = [ self.queue_p[node][mode][ int(frame*self.time_horizon/frames) ] 
                for node in self.graph.get_graph_dic() ]
            scale_set[:, frame] = [ self.queue_p[node][mode][ int(frame*self.time_horizon/frames) ] +100
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


    def passenger_queue_animation_plotly(self, fig, mode, frames):
        
        # color = [ self.queue_p[node][mode][0] for node in self.graph.get_graph_dic() ]
        # scale = [ 16 if (',' in self.graph.get_graph_dic()[node]['mode']) else 12 for node in self.graph.get_graph_dic() ]

        result = np.array([self.queue_p[node][mode] for node in self.queue_p])

        color_set = np.zeros(shape=(len(self.graph.get_graph_dic()), frames))
        scale_set = np.zeros(shape=(len(self.graph.get_graph_dic()), frames))
        for frame_index in range(0, int(frames)):
            color_set[:, frame_index] = [ self.queue_p[node][mode][ int(frame_index*self.time_horizon/frames) ] 
                for node in self.graph.get_graph_dic() ]
            scale_set[:, frame_index] = [ (self.queue_p[node][mode][ int(frame_index*self.time_horizon/frames) ] +self.relativesize)/2
                for node in self.graph.get_graph_dic() ]

        fig = self.plotly_sactter_animation_data(frames=frames, x=self.lon, y=self.lat, color=color_set, scale=scale_set, result=result)
        
        return fig


    def passenger_queue_animation(self, mode, frames, autoplay=False, autosave=False, method='matplotlib'):
        fig = self.graph.plot_topology_edges(self.lon, self.lat, method)
        if (method == 'matplotlib'):
            ani = self.passenger_queue_animation_matplotlib(fig=fig, mode=mode, frames=frames)
        elif (method == 'plotly'):
            ani = self.passenger_queue_animation_plotly(fig=fig, mode=mode, frames=frames)
        
        file_name = 'results/passenger_queue'
        try:
            os.remove(file_name+'.mp4')
        except OSError:
            # logging.warning('Delete log file failed'.format(figs_path))
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
        # print(self.vehicle_queuelen['A'][mode])
        color = [ self.queue_v[node][mode][0] for node in self.graph.get_graph_dic() ]
        scale = [ 300 if (',' in self.graph.get_graph_dic()[node]['mode']) else 100 for node in self.graph.get_graph_dic() ]

        result = np.array([self.queue_v[node][mode] for node in self.queue_v])

        norm = mpl.colors.Normalize(vmin=0, vmax=result.max())

        scat = plt.scatter(x=self.lon, y=self.lat, c=color, s=scale, cmap='Blues', label=color, norm=norm, zorder=2, alpha=0.8, edgecolors='none')
        cbar = plt.colorbar()

        color_set = np.zeros(shape=(len(self.graph.get_graph_dic()), frames))
        for frame_index in range(0, int(frames)):
            color_set[:, frame_index] = [ self.queue_v[node][mode][ int(frame_index*self.time_horizon/frames) ] 
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


    def vehicle_queue_animation_plotly(self, fig, mode, frames):
        color = [ self.queue_v[node][mode][0] for node in self.graph.get_graph_dic() ]
        scale = [ 16 if (',' in self.graph.get_graph_dic()[node]['mode']) else 12 for node in self.graph.get_graph_dic() ]

        result = np.array([self.queue_v[node][mode] for node in self.queue_v])

        color_set = np.zeros(shape=(len(self.graph.get_graph_dic()), frames))
        scale_set = np.zeros(shape=(len(self.graph.get_graph_dic()), frames))
        for frame_index in range(0, int(frames)):
            color_set[:, frame_index] = [ self.queue_v[node][mode][ int(frame_index*self.time_horizon/frames) ] 
                for node in self.graph.get_graph_dic() ]
            scale_set[:, frame_index] = [ (self.queue_v[node][mode][ int(frame_index*self.time_horizon/frames) ] +self.relativesize)/2
                for node in self.graph.get_graph_dic() ]

        fig = self.plotly_sactter_animation_data(frames=frames, x=self.lon, y=self.lat, color=color_set, scale=scale_set, result=result)
        
        return fig


    def vehicle_queue_animation(self, mode, frames, autoplay=False, autosave=False, method='matplotlib'):
        fig = self.graph.plot_topology_edges(self.lon, self.lat, method)

        if (method == 'matplotlib'):
            ani = self.vehicle_queue_animation_matplotlib(fig=fig, mode=mode, frames=frames)
        elif (method == 'plotly'):
            ani = self.vehicle_queue_animation_plotly(fig=fig, mode=mode, frames=frames)

        file_name = 'results/{}_queue'.format(mode)
        try:
            os.remove(file_name+'.mp4')
        except OSError:
            # logging.warning('Delete log file failed'.format(figs_path))
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
        color = [ (self.queue_p[node][mode][0] - self.queue_v[node][mode][0]) 
            for node in self.graph.get_graph_dic() ]
        scale = [ 300 if (',' in self.graph.get_graph_dic()[node]['mode']) else 100 for node in self.graph.get_graph_dic() ]

        result = np.array([ (self.queue_p[node][mode] - self.queue_v[node][mode])
            for node in self.queue_v ])

        # norm = mpl.colors.Normalize(vmin=result.min(), vcenter=0, vmax=result.max())
        norm = MidpointNormalize(vmin=result.min(), vcenter=0, vmax=result.max())

        scat = plt.scatter(x=self.lon, y=self.lat, c=color, s=scale, cmap='coolwarm', label=color, norm=norm, zorder=2, alpha=0.8, edgecolors='none')
        cbar = plt.colorbar()
        color_set = np.zeros(shape=(len(self.graph.get_graph_dic()), frames))
        for frame_index in range(0, int(frames)):
            index = int(frame_index*self.time_horizon/frames)
            color_set[:, frame_index] = [ (self.queue_p[node][mode][index] - self.queue_v[node][mode][index]) 
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


    def combination_queue_animation_plotly(self, fig, mode, frames):
        color = [ (self.queue_p[node][mode][0] - self.queue_v[node][mode][0]) 
            for node in self.graph.get_graph_dic() ]
        scale = [ 16 if (',' in self.graph.get_graph_dic()[node]['mode']) else 12 for node in self.graph.get_graph_dic() ]

        result = np.array([ (self.queue_p[node][mode] - self.queue_v[node][mode])
            for node in self.queue_v ])

        color_set = np.zeros(shape=(len(self.graph.get_graph_dic()), frames))
        scale_set = np.zeros(shape=(len(self.graph.get_graph_dic()), frames))
        for frame_index in range(0, int(frames)):
            index = int(frame_index*self.time_horizon/frames)
            color_set[:, frame_index] = [ (self.queue_p[node][mode][index] - self.queue_v[node][mode][index]) 
                for node in self.graph.get_graph_dic() ]
            scale_set[:, frame_index] = [ (np.abs(self.queue_p[node][mode][index] - self.queue_v[node][mode][index]) +self.relativesize)/2
                for node in self.graph.get_graph_dic() ]            

        fig = self.plotly_sactter_animation_data(frames=frames, x=self.lon, y=self.lat, color=color_set, scale=scale_set, result=result)
        
        return fig

    '''
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
            # data = go.Mesh3d(x=X, y=Y, z=z[:, frame_index], i=I, j=J, k=K, color="#ba2461", flatshading=True)
            # print(data)
            # data_list.append(data)
            data = {'X': X, 'Y': Y, 'Z': Z, 'I': I, 'J': J, 'K': K}
            data_list.append(data)

        fig = self.plotly_3d_animation_data(frames=frames, data=data_list)
        # layout = go.Layout(width=1200, height=900, title_text='3D Bar Chart', title_x =0.5)

        # fig = go.Figure(data=[mesh3d], layout=layout)
        # pt.offline.plot(fig)    
        return fig
    '''


    def combination_queue_animation(self, mode, frames, autoplay=False, autosave=False, method='matplotlib'):
        self.lat = [ self.graph.get_node_location(node)[0] for node in self.graph.get_graph_dic() ]
        self.lon = [ self.graph.get_node_location(node)[1] for node in self.graph.get_graph_dic() ]

        fig = self.graph.plot_topology_edges(self.lon, self.lat, method)

        if (method == 'matplotlib'):
            ani = self.combination_queue_animation_matplotlib(fig=fig, mode=mode, frames=frames)
        elif (method == 'plotly'):
            ani = self.combination_queue_animation_plotly(fig=fig, mode=mode, frames=frames)
        '''
        elif (method == 'plotly_3d'):
            ani = self.combination_queue_animation_plotly_3d(fig=fig, x=self.lon, y=self.lat, mode=mode, frames=frames)
        '''

        file_name = 'results/{}_combined_queue'.format(mode)
        try:
            os.remove(file_name+'.mp4')
        except OSError:
            # logging.warning('Delete log file failed'.format(figs_path))
            pass
        
        if (autosave and method == 'matplotlib'):
            ani.save(file_name+'.mp4', fps=12, dpi=300)

        print('Done')
        # animation.to_html5_video()
        if (autoplay):
            if (method == 'matplotlib'):
                plt.show()
            elif (method == 'plotly' or method == 'plotly_3d'):
                pt.offline.plot(ani, filename=file_name+'.html')