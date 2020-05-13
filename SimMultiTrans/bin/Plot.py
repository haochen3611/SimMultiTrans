#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
from datetime import datetime, timedelta
import logging

import matplotlib as mpl
import numpy as np
import plotly as pt
import plotly.graph_objects as go

from SimMultiTrans.utils import CONFIG, RESULTS

logger = logging.getLogger(__name__)


class Plot(object):

    def __init__(self, graph, time_horizon=None, start_time=None):
        self.graph = graph
        self.time_horizon = time_horizon
        self.start_time = start_time
        self.lat = np.asarray([self.graph.graph_top[node]['lat'] for node in self.graph.graph_top])
        self.lon = np.asarray([self.graph.graph_top[node]['lon'] for node in self.graph.graph_top])

        self.relativesize = 120
        self.basicsize = 6

        try:
            self.mapbox_access_token = open(os.path.join(CONFIG, '.mapbox_token')).read()
            self.map_style = open(os.path.join(CONFIG, '.mapbox_style')).read()
        except OSError:
            logger.error('Map Key Error!')
            raise

    def import_results(self, path_name=RESULTS):
        self.graph.import_graph(os.path.join(path_name, 'city_topology.json'))

        with open(os.path.join(path_name, 'simulation_info.json')) as json_file:
            simulation_info = json.load(json_file)

        self.time_horizon = simulation_info['Time_horizon']
        self.start_time = datetime.strptime(simulation_info['Start_time'], "%H:%M:%S")
        self.vehicle_attri = simulation_info['Vehicle']
        self.reb_method = simulation_info['Rebalancing_method']
        self.routing_method = simulation_info['Routing_method']
        self.duration = simulation_info['Duration']

        self.lat = np.asarray([self.graph.graph_top[node]['lat'] for node in self.graph.graph_top])
        self.lon = np.asarray([self.graph.graph_top[node]['lon'] for node in self.graph.graph_top])

        with open(os.path.join(path_name, 'passenger_queue.json')) as json_file:
            queue_p = json.load(json_file)

        with open(os.path.join(path_name, 'vehicle_queue.json')) as json_file:
            queue_v = json.load(json_file)

        with open(os.path.join(path_name, 'wait_time.json')) as json_file:
            waittime = json.load(json_file)

        with open(os.path.join(path_name, 'metrics.json')) as json_file:
            metrics = json.load(json_file)

        self.total_trip = metrics['total_trip']
        self.total_tripdist = metrics['total_tripdist']
        self.total_triptime = metrics['total_triptime']
        self.total_arrival = metrics['total_arrival']
        self.sum_totalarrival = metrics['total_num_arrival']
        self.not_served = metrics['not_served']
        # self.not_served = 0


        for node in self.graph.graph_top:
            for mode in self.vehicle_attri:
                queue_p[node][mode] = np.array(queue_p[node][mode])
                queue_v[node][mode] = np.array(queue_v[node][mode])
                waittime[node][mode] = np.array(waittime[node][mode])

        self.queue_p = queue_p
        self.queue_v = queue_v
        self.waittime_p = waittime

    def set_plot_theme(self, theme):
        if theme in ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]:
            pt.io.templates.default = theme
        else:
            pt.io.templates.default = 'plotly'

    def plot_passenger_queuelen(self, mode, time):
        time_step = (datetime.strptime(time, '%H:%M:%S') - self.start_time).seconds

        if time_step < 0 or time_step > self.time_horizon:
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
                lat=np.mean(self.lat),
                lon=np.mean(self.lon)
            ),
            'pitch': 60,
            'zoom': 11,
            'style': self.map_style
        }

        data = np.array([self.queue_v[node][mode][time_step] for node in self.graph.graph_top])
        cmin = np.min(data.min(), 0)
        cmax = np.max(data.max(), 0)
        cmax = cmin + 1 if (cmax - cmin == 0) else cmax
        colorsacle = [[0, '#33691E'], [np.abs(cmin) / (cmax - cmin), '#FAFAFA'], [1, '#FF6F00']]
        sizescale = self.relativesize / np.max([cmax, np.abs(cmin)])
        text_str = [f'{self.graph.get_all_nodes()[index]}: {data[index]}' for index in range(len(data))]
        data_dict = {
            'type': 'scattermapbox',
            'lon': self.lon, 'lat': self.lat,
            'mode': 'markers',
            'name': 'Queue',
            'text': text_str,
            'marker': {
                'size': data * sizescale + self.basicsize,
                # 'size': np.log(data+self.relativesize),
                'color': data, 'colorscale': colorsacle,
                'cmin': data.min(), 'cmax': data.max(), 'colorbar': dict(title='Queue')
            }
        }
        fig_dict['data'].append(data_dict)
        fig = go.Figure(fig_dict)

        file_name = os.path.join(RESULTS, f'Passenger_{mode}_queue_at_{time}')
        # fig.update_layout(template='plotly_dark')
        pt.offline.plot(fig, filename=file_name + '.html', auto_open=False)

    def plotly_sactter_animation_data(self, frames, lon, lat, data):
        ani_dict = {'data': [], 'layout': {}, 'frames': []}

        # fill in most of layout
        ani_dict['layout']['xaxis'] = {'title': 'Latitude'}
        ani_dict['layout']['yaxis'] = {'title': 'Longitude'}
        ani_dict['layout']['hovermode'] = 'closest'
        ani_dict['layout']['mapbox'] = {
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
        '''
        ani_dict['layout']['sliders'] = {
            'args': [ 'transition', { 'duration': 400, 'easing': 'cubic-in-out' } ],
            'initialValue': '0', 'plotlycommand': 'animate', 'values': range(frames), 'visible': True
        }
        '''
        ani_dict['layout']['updatemenus'] = [{
            'buttons': [
                {'args': [None, {'frame': {'duration': 500, 'redraw': True},
                                 'fromcurrent': True, 'transition': {'duration': 300, 'easing': 'quadratic-in-out'}}],
                 'label': 'Play', 'method': 'animate'},
                {'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate',
                                   'transition': {'duration': 0}}],
                 'label': 'Pause', 'method': 'animate'}
            ],
            'direction': 'left', 'pad': {'r': 10, 't': 87},
            'showactive': False, 'type': 'buttons', 'x': 0.1, 'xanchor': 'right', 'y': 0, 'yanchor': 'top'
        }]

        sliders_dict = {'active': 0, 'yanchor': 'top', 'xanchor': 'left',
                        'currentvalue': {'font': {'size': 20}, 'prefix': 'Time:', 'visible': True, 'xanchor': 'right'},
                        'transition': {'duration': 300, 'easing': 'cubic-in-out'}, 'pad': {'b': 10, 't': 50},
                        'len': 0.9, 'x': 0.1, 'y': 0, 'steps': []}

        # make data
        time = 0
        # colorsacle = 'OrRd' if (result.min() == 0) else 'balance'
        # set 0 be white
        cmin = np.min([data.min(), 0])
        cmax = np.max([data.max(), 0])
        cmax = cmin + 1 if (cmax - cmin == 0) else cmax
        colorsacle = [[0, '#33691E'], [np.abs(cmin) / (cmax - cmin), '#FAFAFA'], [1, '#FF6F00']]
        sizescale = self.relativesize / np.max([cmax, np.abs(cmin)])
        text_str = [f'{self.graph.get_all_nodes()[index]}: {data[index, 0]}' for index in range(len(data))]
        data_dict = {
            'type': 'scattermapbox',
            'lon': lon, 'lat': lat,
            'mode': 'markers',
            'name': 'Queue',
            'text': text_str,
            'marker': {  # 'size': np.abs(data[:, 0])+self.relativesize,
                'size': np.abs(data[:, 0]) * sizescale + self.basicsize,
                'color': data[:, 0], 'colorscale': colorsacle,
                'cmin': cmin, 'cmax': cmax, 'colorbar': dict(title='Queue')}
        }
        ani_dict['data'].append(data_dict)

        # make frames
        for frame_index in range(frames):
            frame = {'data': [], 'name': str(frame_index)}
            text_str = [f'{self.graph.get_all_nodes()[index]}: {data[index, frame_index]}' for index in range(len(data))]
            data_dict = {
                'type': 'scattermapbox',
                'lon': lon, 'lat': lat,
                'mode': 'markers',
                'name': 'Queue',
                'text': text_str,
                'marker': {
                    'size': np.abs(data[:, frame_index]) * sizescale + self.basicsize,
                    # 'size': 5*np.log(np.abs(data[:, frame_index])+self.relativesize),
                    'color': data[:, frame_index], 'colorscale': colorsacle,
                    'cmin': cmin, 'cmax': cmax, 'colorbar': dict(title='Queue')}
            }

            frame['data'].append(data_dict)

            ani_dict['frames'].append(frame)
            frame_time = timedelta(seconds=frame_index * self.time_horizon / frames) + self.start_time
            slider_step = {'args': [
                [frame_index],
                {'frame': {'duration': 300, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 300}}],
                'label': frame_time.strftime('%H:%M'), 'method': 'animate'}
            sliders_dict['steps'].append(slider_step)

        ani_dict['layout']['sliders'] = [sliders_dict]

        # fig = go.Figure(fig_dict)
        # fig.update_layout(template='plotly_dark')
        return ani_dict

    def passenger_queue_animation(self, mode, frames):
        logger.info(f'Plot queue length of passengers who take {mode}')
        data = np.zeros(shape=(len(self.graph.graph_top), frames))
        for frame_index in range(0, int(frames)):
            data[:, frame_index] = [self.queue_p[node][mode][int(frame_index * self.time_horizon / frames)]
                                    for node in self.graph.graph_top]

        ani_dict = self.plotly_sactter_animation_data(frames=frames, lon=self.lon, lat=self.lat, data=data)
        ani = go.Figure(ani_dict)
        pt.offline.plot(ani, filename=os.path.join(RESULTS, 'passenger_queue.html'), auto_open=False)

    def vehicle_queue_animation(self, mode, frames):
        logger.info(f'Plot queue length of {mode}')
        data = np.zeros(shape=(len(self.graph.graph_top), frames))
        for frame_index in range(0, int(frames)):
            data[:, frame_index] = [self.queue_v[node][mode][int(frame_index * self.time_horizon / frames)]
                                    for node in self.graph.graph_top]

        ani_dict = self.plotly_sactter_animation_data(frames=frames, lon=self.lon, lat=self.lat, data=data)
        ani = go.Figure(ani_dict)
        pt.offline.plot(ani, filename=os.path.join(RESULTS, f'{mode}_queue.html'), auto_open=False)

    def combination_queue_animation(self, mode, frames, suffix=None):
        logger.info(f'Plot combined queue length of passengers and {mode}')
        if suffix is None:
            suffix = ''
        else:
            suffix = "_" + suffix

        self.lat = [self.graph.graph_top[node]['lat'] for node in self.graph.graph_top]
        self.lon = [self.graph.graph_top[node]['lon'] for node in self.graph.graph_top]

        data = np.zeros(shape=(len(self.graph.graph_top), frames))
        # sum all buses
        bus_list = []
        if mode == 'bus':
            for node in self.graph.graph_top:
                modelist = self.graph.graph_top[node]['mode'].split(',')
                modelist = [bus for bus in modelist if ('BUS' in bus)]
                # print(modelist)
                bus_list = list(set(bus_list + modelist))
            # print(bus_list)
            for bus in bus_list:
                for frame_index in range(0, int(frames)):
                    index = int(frame_index * self.time_horizon / frames)
                    data[:, frame_index] = data[:, frame_index] + [
                        (self.queue_p[node][bus][index] - self.queue_v[node][bus][index])
                        for node in self.graph.graph_top]
        else:
            for frame_index in range(0, int(frames)):
                index = int(frame_index * self.time_horizon / frames)
                data[:, frame_index] = [(self.queue_p[node][mode][index] - self.queue_v[node][mode][index])
                                        for node in self.graph.graph_top]

        ani_dict = self.plotly_sactter_animation_data(frames=frames, lon=self.lon, lat=self.lat, data=data)
        ani = go.Figure(ani_dict)

        pt.offline.plot(ani, filename=os.path.join(RESULTS, f'{mode}_combined_queue{suffix}.html'), auto_open=False)

    def plot_passenger_queuelen_time(self, mode, suffix=None):
        if suffix is None:
            suffix = ''
        else:
            suffix = "_" + suffix
        fig_dict = {'data': [], 'layout': {}}
        fig_dict['layout']['xaxis'] = {'title': 'Time'}
        fig_dict['layout']['xaxis']['ticktext'] = [
            (timedelta(seconds=t) + self.start_time).strftime('%H:%M:%S') for t in
            range(0, self.time_horizon, int(self.time_horizon / 10))
        ]
        fig_dict['layout']['xaxis']['tickvals'] = [t for t in range(0, self.time_horizon, int(self.time_horizon / 10))]
        fig_dict['layout']['yaxis'] = {'title': f'Imbalance: # passengers - # {mode}'}
        fig_dict['layout']['hovermode'] = 'closest'
        fig_dict['layout']['title'] = 'Changing of Imbalance'

        # fig = go.Figure()
        x = np.arange(start=0, stop=self.time_horizon, step=1)

        for index, node in enumerate(self.queue_p):
            # does not plot the node only with walk
            if self.graph.graph_top[node]['mode'] != 'walk':
                y = self.queue_p[node][mode] - self.queue_v[node][mode]
                data_dict = {
                    'type': 'scatter', 'x': x, 'y': y, 'name': node
                }
                fig_dict['data'].append(data_dict)

        fig = go.Figure(fig_dict)

        # fig.update_layout(template='plotly_dark')
        pt.offline.plot(fig, filename=os.path.join(RESULTS, f'{mode}_queue_time{suffix}.html'), auto_open=False)

    def plot_passenger_waittime(self, mode, suffix=None):
        if suffix is None:
            suffix = ''
        else:
            suffix = "_" + suffix
        fig_dict = {'data': [], 'layout': {}}
        fig_dict['layout']['xaxis'] = {'title': 'Region ID'}
        fig_dict['layout']['xaxis']['type'] = 'category'
        fig_dict['layout']['yaxis'] = {'title': 'Waiting Time (s)'}
        fig_dict['layout']['hovermode'] = 'closest'
        fig_dict['layout']['title'] = 'Average Waiting Time'

        x = [node for node in self.graph.get_all_nodes() if (self.graph.graph_top[node]['mode'] != 'walk')]
        # print(self.passenger_waittime)
        y = [self.waittime_p[node][mode] for node in x]
        data_dict = {
            'type': 'bar', 'x': x, 'y': y, 'marker_color': 'indianred', 'text': [f'{m} min' for m in y],
        }
        fig_dict['data'].append(data_dict)
        fig = go.Figure(fig_dict)

        pt.offline.plot(fig, filename=os.path.join(RESULTS, f'{mode}_waittime{suffix}.html'), auto_open=False)

    def plotly_metrics_data(self, mode):
        # fig = pt.subplots.make_subplots(rows=2, cols=2)
        fig_dict = {'data': [], 'layout': {}}

        fig_dict['layout']['hovermode'] = 'closest'
        fig_dict['layout']['title'] = f'Operational Metrics of {mode}'
        fig_dict['layout']['titlefont'] = {'size': 24}
        fig_dict['layout']['legend'] = {'x': 0, 'y': 0.4}

        # average waiting time
        x = [node for node in self.graph.get_all_nodes() if (self.graph.graph_top[node]['mode'] != 'walk')]
        # print(self.passenger_waittime)
        wait_y = np.around([self.waittime_p[node][mode] / 60.0 for node in x], decimals=2)
        data_dict = {
            'type': 'bar', 'x': x, 'y': wait_y,
            'name': 'Waiting Time', 'offsetgroup': '0', 'text': [f'{m} min' for m in wait_y],
            'hovertemplate': 'Region ID: %{x}' + '<br>Waiting Time: %{text}<br>' + '<extra></extra>',
            'marker_color': 'indianred', 'xaxis': 'x', 'yaxis': 'y'
        }
        fig_dict['data'].append(data_dict)
        fig_dict['layout']['xaxis'] = {
            'title': 'Region ID', 'type': 'category', 'domain': [0, 1], 'titlefont': {'size': 16}
        }
        fig_dict['layout']['yaxis'] = {
            'title': 'Averaged Waiting Time (min)', 'gridcolor': '#FBE9E7',
            'titlefont': {'size': 16, 'color': 'indianred'}, 'tickfont': {'color': 'indianred'},
            'domain': [0, 0.45], 'anchor': 'x', 'overlaying': 'y2', 'zerolinecolor': '#E0E0E0'
        }

        # passenger throughtput
        thrpt_y = np.around([
            (self.total_trip['total'][node][mode] - self.total_trip['reb'][node][mode]) / float(
                self.time_horizon) * 60
            for node in x
        ], decimals=2)
        data_dict = {
            'type': 'bar', 'x': x, 'y': thrpt_y,
            'name': 'Throughput', 'offsetgroup': '1', 'text': [f'{m} {mode} per min' for m in thrpt_y],
            'hovertemplate': 'Region ID: %{x}' + '<br>Throughput: %{text}<br>' + '<extra></extra>',
            'marker_color': 'lightsalmon', 'xaxis': 'x', 'yaxis': 'y2'
        }
        fig_dict['data'].append(data_dict)

        fig_dict['layout']['yaxis2'] = {
            'title': 'Throughput (#/min)', 'gridcolor': '#FFF3E0',
            'titlefont': {'size': 16, 'color': 'lightsalmon'}, 'tickfont': {'color': 'lightsalmon'},
            'domain': [0, 0.45], 'anchor': 'x', 'side': 'right', 'zerolinecolor': '#E0E0E0'
        }

        # trip time/dist
        # x = ['Trip Time', 'Trip Distance']
        sum_trip = {'total': 0, 'reb': 0}
        for node in x:
            sum_trip['total'] += self.total_trip['total'][node][mode]
            sum_trip['reb'] += self.total_trip['reb'][node][mode]

        x = ['Riding', 'Rebalancing']
        if sum_trip['reb'] == 0:
            # y = np.around([self.total_tripdist['total'][mode]/float(sum_trip['total'])/1609.34, 0], decimals=2)
            y = np.around([self.total_tripdist['total'][mode] / 1609.34, 0], decimals=2)
        else:
            y = np.around([
                (self.total_tripdist['total'][mode] - self.total_tripdist['reb'][mode]) / 1609.34,
                self.total_tripdist['reb'][mode] / 1609.34
            ], decimals=2)
            '''
            y = np.around([
                (self.total_tripdist['total'][mode]-self.total_tripdist['reb'][mode])/float(sum_trip['total']-sum_trip['reb'])/1609.34, 
                self.total_tripdist['reb'][mode]/float(sum_trip['reb'])/1609.34
            ], decimals=2)
            '''
        color = ['#80DEEA', '#0097A7']
        data_dict = {
            'title': 'Total Trip Distance (mile)', 'titlefont': {'size': 16}, 'titleposition': 'bottom center',
            'type': 'pie', 'labels': x, 'values': y,
            'textposition': 'inside', 'textinfo': 'percent+label', 'textfont_size': 16,
            'name': 'Total Trip Distance',
            'hovertemplate': 'Averaged %{label} Distance: ' + '%{value} miles' + '<extra></extra>',
            'marker_colors': color,
            'domain': {'x': [0.55, 0.75], 'y': [0.55, 1]}, 'showlegend': False
        }
        fig_dict['data'].append(data_dict)

        x = ['Riding', 'Rebalancing']
        if sum_trip['reb'] == 0:
            # y = np.around([self.total_triptime['total'][mode]/float(sum_trip['total'])/1609.34, 0], decimals=2)
            y = np.around([self.total_triptime['total'][mode] / 1609.34, 0], decimals=2)
        else:
            y = np.around([
                (self.total_triptime['total'][mode] - self.total_triptime['reb'][mode]) / 3600.0,
                self.total_triptime['reb'][mode] / 3600.0
            ], decimals=2)
            '''
            y = np.around([
                (self.total_triptime['total'][mode]-self.total_triptime['reb'][mode])/float(sum_trip['total']-sum_trip['reb'])/3600.0, 
                self.total_triptime['reb'][mode]/float(sum_trip['reb'])/3600.0
            ], decimals=2)
            '''
        color = ['#81D4FA', '#0288D1']
        data_dict = {
            'title': 'Total Trip Time (hour)', 'titlefont': {'size': 16}, 'titleposition': 'bottom center',
            'type': 'pie', 'labels': x, 'values': y,
            'textposition': 'inside', 'textinfo': 'percent+label', 'textfont_size': 16,
            'name': 'Total Trip Time',
            'hovertemplate': 'Averaged %{label} Time: ' + '%{value} hours' + '<extra></extra>',
            'marker_colors': color,
            'domain': {'x': [0.8, 1], 'y': [0.55, 1]}, 'showlegend': False
        }
        fig_dict['data'].append(data_dict)

        # general metrics
        # total trips
        '''
        x = ['Total Trips']
        y = [sum_trip['total']]
        data_dict = { 
            'type':'bar', 'x': x, 'y': y, 'name': 'Total Trips', 'text': y,
            'textposition': 'outside',
            'hovertemplate': 'Total %{y} Trips'+'<extra></extra>',
            'marker_color': '#D4E157', 'xaxis': 'x2', 'yaxis': 'y3', 'showlegend': False
        }
        fig_dict['data'].append(data_dict)
        fig_dict['layout']['xaxis2'] = {
            'title': 'General Metrics', 'type': 'category', 'domain': [0, 0.45], 'anchor': 'y3',
            'titlefont': {'size': 16}, 'tickfont': {'size': 14}
        }
        fig_dict['layout']['yaxis3'] = {
            'title': '', 'showgrid': False, 'ticks': '', 'showticklabels': False, 'showline': False,
            'domain': [0.6, 1], 'anchor': 'x2', 'overlaying': 'y5', 'range': [0, y[0]*5.4], 'zerolinecolor': '#E0E0E0'
        }

        x = ['Total Miles Traveled']
        y = np.around([self.total_tripdist['total'][mode]/1609.34], decimals=2)
        data_dict = { 
            'type':'bar', 'x': x, 'y': y, 'name': 'Total Miles Traveled', 'text': [f'{m} miles' for m in y], 
            'textposition': 'outside',
            'hovertemplate': 'Total %{y} Miles Traveled'+'<extra></extra>',
            'marker_color': '#9CCC65', 'xaxis': 'x2', 'yaxis': 'y4', 'showlegend': False
        }
        fig_dict['data'].append(data_dict)
        fig_dict['layout']['yaxis4'] = {
            'title': '', 'showgrid': False, 'ticks': '', 'showticklabels': False, 'showline': False,
            'domain': [0.6, 1], 'anchor': 'x2', 'overlaying': 'y5', 'range': [0, y[0]*1.4], 'zerolinecolor': '#E0E0E0'
        }

        x = ['Total Hours Traveled']
        y = np.around([self.total_tripdist['total'][mode]/3600.0],decimals=2)
        data_dict = { 
            'type':'bar', 'x': x, 'y': y, 'name': 'Total Hours Traveled', 'text': [f'{m} hours' for m in y],  
            'textposition': 'outside',
            'hovertemplate': 'Total %{y} Hours Traveled'+'<extra></extra>',
            'marker_color': '#66BB6A', 'xaxis': 'x2', 'yaxis': 'y5', 'showlegend': False
        }
        fig_dict['data'].append(data_dict)
        fig_dict['layout']['yaxis5'] = {
            'title': '', 'showgrid': False, 'ticks': '', 'showticklabels': False, 'showline': False,
            'domain': [0.6, 1], 'anchor': 'x2', 'range': [0, y[0]*2.7], 'zerolinecolor': '#E0E0E0'
        }
        '''
        # metrics table
        header = {
            'font_size': 16,
            'align': 'center',
            'values': ['Metric', 'Total', 'Rebalancing', 'Variance']
        }
        reb_div = float(sum_trip['total']) if sum_trip['total'] != 0 else 1.0

        wait_y = np.array([y for y in wait_y if y != 0])
        thrpt_y = np.array([y for y in thrpt_y if y != 0])

        cells = {
            'font_size': 14,
            'height': 40,
            'align': 'center',
            'values': [
                ['Arrivals', 'Trips', 'Averaged Miles Traveled', 'Averaged Minutes Traveled', 'Averaged Waiting Time',
                 'Average Throughput'],
                [
                    f'{self.sum_totalarrival} ({self.not_served})',
                    sum_trip['total'],
                    np.around(self.total_tripdist['total'][mode] / float(sum_trip['total']) / 1609.34, decimals=2),
                    np.around(self.total_triptime['total'][mode] / float(sum_trip['total']) / 60.0, decimals=2),
                    np.around(np.mean(wait_y), decimals=2),
                    np.around(np.mean(thrpt_y), decimals=2),
                ],
                [
                    'N/A',
                    sum_trip['reb'],
                    np.around(self.total_tripdist['reb'][mode] / reb_div / 1609.34, decimals=2),
                    np.around(self.total_triptime['reb'][mode] / reb_div / 60.0, decimals=2),
                    'N/A',
                    'N/A'
                ],
                [
                    'N/A',
                    'N/A',
                    'N/A',
                    'N/A',
                    np.around(np.var(wait_y), decimals=2),
                    np.around(np.var(thrpt_y), decimals=2),
                ]
            ]
        }
        data_dict = {
            # 'title': 'Average Trip Time (hour)', 'titlefont': {'size': 16}, 'titleposition': 'bottom center',
            'type': 'table', 'header': header, 'cells': cells, 'name': 'Table',
            'domain': {'x': [0, 0.5], 'y': [0.55, 1]}
        }
        fig_dict['data'].append(data_dict)

        return fig_dict

    def plot_metrics(self, mode):
        fig_dict = self.plotly_metrics_data(mode)
        fig = go.Figure(fig_dict)

        # fig.update_layout(template='plotly_dark')
        pt.offline.plot(fig, filename=os.path.join(RESULTS, f'{mode}_metrics.html'), auto_open=False)

    def plotly_metrics_animation_data(self, mode, policies):
        ani_dict = {'data': [], 'layout': {}, 'frames': []}

        fig_dict_set = []
        for policy in policies:
            self.import_results(f'results/{policy}')
            fig_dict = self.plotly_metrics_data(mode)
            fig_dict_set.append(fig_dict)

        ani_dict['data'] = fig_dict_set[0]['data']
        ani_dict['layout'] = fig_dict_set[0]['layout']

        ani_dict['layout']['updatemenus'] = [{
            'direction': 'left', 'pad': {'r': 10, 't': 87},
            'showactive': False, 'type': 'buttons', 'x': 0.1, 'xanchor': 'right', 'y': 0, 'yanchor': 'top'
        }]

        sliders_dict = {'active': 0, 'yanchor': 'top', 'xanchor': 'left',
                        'currentvalue': {'font': {'size': 20}, 'prefix': 'Rebalancing Policy: ', 'visible': True,
                                         'xanchor': 'right'},
                        'transition': {'duration': 300, 'easing': 'cubic-in-out'}, 'pad': {'b': 10, 't': 50},
                        'len': 0.9, 'x': 0.1, 'y': 0, 'steps': []}

        v_max = {
            'Waiting Time': [0, 'yaxis'],
            'Throughput': [0, 'yaxis2']
        }

        # uniform yaxis
        for frame_index, fig_dict in enumerate(fig_dict_set):
            for data_dict in fig_dict['data']:
                if data_dict['name'] in v_max:
                    v_max[data_dict['name']][0] = max(v_max[data_dict['name']][0], max(data_dict['y']))

        # make frames
        for frame_index, fig_dict in enumerate(fig_dict_set):
            frame = {'data': [], 'name': str(frame_index)}
            # text_str = [f'{self.graph.get_all_nodes()[index]}: {data[index, frame_index]}' for index in range(len(data))]
            for data_dict in fig_dict['data']:
                if data_dict['name'] in v_max:
                    fig_dict['layout'][v_max[data_dict['name']][1]]['range'] = [0, v_max[data_dict['name']][0] * 1.1]

            frame['data'] = fig_dict['data']

            ani_dict['frames'].append(frame)

            slider_step = {'args': [
                [frame_index],
                {'frame': {'duration': 300, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 300}}],
                'label': policies[frame_index].split('__')[0], 'method': 'animate'}
            sliders_dict['steps'].append(slider_step)

        ani_dict['layout']['sliders'] = [sliders_dict]

        return ani_dict

    def plot_metrics_animation(self, mode, policies):
        ani_dict = self.plotly_metrics_animation_data(mode, policies)
        ani = go.Figure(ani_dict)
        pt.offline.plot(ani, filename=os.path.join(RESULTS, f'{mode}_metrics_comparison.html'), auto_open=False)

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
                lat=np.mean(self.lat),
                lon=np.mean(self.lon)
            ),
            'pitch': 60,
            'zoom': 11,
            'style': self.map_style
        }

        size = np.zeros([len(self.lon), 1]) + 10
        color = ['#FAFAFA' for node in self.lon]

        text_str = [f'{self.graph.get_all_nodes()[index]}' for index in range(len(self.lon))]
        data_dict = {
            'type': 'scattermapbox',
            'lon': self.lon, 'lat': self.lat,
            'mode': 'markers',
            'name': 'Queue',
            'text': text_str,
            'marker': {'size': size, 'color': color}
        }
        fig_dict['data'].append(data_dict)
        fig = go.Figure(fig_dict)

        # fig.update_layout(template='plotly_dark')
        pt.offline.plot(fig, filename=os.path.join(RESULTS, 'Topology.html'), auto_open=False)


class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
