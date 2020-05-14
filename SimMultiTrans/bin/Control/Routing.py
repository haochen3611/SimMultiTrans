#!/usr/bin/env python
# -*- coding: utf-8 -*-
# from Node import Node

import numpy as np

import random
import copy
import logging


logger = logging.getLogger(__name__)


class Routing(object):
    def __init__(self, graph, vehicle_attri):
        self.graph = graph
        self.path = {
            'bus_simplex': {},
            'simplex': {},
            'bus_walk_simplex': {},
            'taxi_walk_simplex': {}
        }
        self.vehicle_attri = vehicle_attri
        self.routing_method = 'simplex'
        self.near_nei = {}
        self.info = None

    def syn_info(self, info):
        self.info = info

    def get_methods(self):
        return list(self.path.keys())

    def set_routing_method(self, method):
        if method in self.path:
            self.routing_method = method
            logger.info(f'Routing Method: {method}')
        else:
            logger.info(f'{method} is unavailable. Will use {self.routing_method}.')

    def get_path(self, ori, dest):
        if ori not in self.graph.graph_top and dest not in self.graph.graph_top:
            logger.warning('invalid path')
            return {}

        if ori in self.path[self.routing_method] and dest in self.path[self.routing_method][ori]:
            if self.path[self.routing_method][ori][dest]:
                # print(self.path[self.routing_method][ori][dest])
                # path = self.path[self.routing_method][ori][dest]
                # print(path)
                return self.path[self.routing_method][ori][dest]

        method_set = {
            'bus_simplex': self.bus_simplex,
            'simplex': self.simplex,
            'bus_walk_simplex': self.bus_walk_simplex,
            'taxi_walk_simplex': self.taxi_walk_simplex
        }
        # route = Routing(self, ori, dest)
        # print('routing: ', ori, dest)
        path = method_set[self.routing_method](ori, dest)
        # print(path)
        self.save_path(ori, dest, self.routing_method, path)
        return path

    def simplex(self, ori, dest):
        if dest in self.graph.graph_top[ori]['nei']:
            return {ori: {'dest': dest, 'info': self.pathinfo_generator(ori=ori, dest=dest, method='simplex')}}
        else:
            # print('problem')
            return {}

    def bus_walk_simplex(self, ori, dest):
        # stops = ori
        path = {}

        # check avialability
        ori_nei_id = [node for node in self.graph.graph_top[ori]['nei']]
        ori_nei_dist = np.array(
            [self.graph.graph_top[ori]['nei'][node]['dist'] for node in self.graph.graph_top[ori]['nei']])

        # print(ori_nei)
        dest_nei_id = [node for node in self.graph.graph_top[dest]['nei']]
        dest_nei_dist = np.array(
            [self.graph.graph_top[dest]['nei'][node]['dist'] for node in self.graph.graph_top[dest]['nei']])

        # print(dest_nei)
        # print(np.where(ori_nei_dist == np.amin(ori_nei_dist))[0][0])
        # print(np.where(dest_nei_dist == np.amin(dest_nei_dist))[0][0])

        ori_trans = ori_nei_id[
            np.where(ori_nei_dist == np.amin(ori_nei_dist[
                                                 ori_nei_dist != np.amin(ori_nei_dist)
                                                 ]))[0][0]
        ]

        dest_trans = dest_nei_id[
            np.where(dest_nei_dist == np.amin(dest_nei_dist[
                                                  dest_nei_dist != np.amin(dest_nei_dist)
                                                  ]))[0][0]
        ]

        ori_trans_modes = self.graph.graph_top[ori_trans]['mode'].split(',')
        dest_trans_modes = self.graph.graph_top[dest_trans]['mode'].split(',')
        mode = list(set(ori_trans_modes).intersection(dest_trans_modes))
        # no bus available, take taxi
        # also add some randomness
        if len(mode) > 2 and random.choice([True, False]):
            path.update(
                {ori: {'dest': ori_trans, 'info': self.pathinfo_generator(ori=ori, dest=ori_trans, method='walk')}})

            if ori_trans != dest_trans:
                # walk to destination
                path.update({ori_trans: {'dest': dest_trans,
                                         'info': self.pathinfo_generator(ori=ori_trans, dest=dest_trans,
                                                                         method=mode[0])}})

            path.update(
                {dest_trans: {'dest': dest, 'info': self.pathinfo_generator(ori=dest_trans, dest=dest, method='walk')}})
            # return path
            # print(ori, dest)
            # print('a', path)
        # bus available
        else:
            path.update(
                {ori: {'dest': ori_trans, 'info': self.pathinfo_generator(ori=ori, dest=ori_trans, method='walk')}})
            if ori_trans != dest_trans:
                # walk to destination
                path.update({ori_trans: {'dest': dest_trans,
                                         'info': self.pathinfo_generator(ori=ori_trans, dest=dest_trans,
                                                                         method='taxi')}})

            path.update(
                {dest_trans: {'dest': dest, 'info': self.pathinfo_generator(ori=dest_trans, dest=dest, method='walk')}})
            # print(ori, dest)
            # print('b', path)
        # print(path)
        return path

    def taxi_walk_simplex(self, ori, dest):
        # stops = ori
        path = {}

        # check avialability
        ori_nei_id = np.array([node for node in self.graph.graph_top[ori]['nei']])
        ori_nei_dist = np.array(
            [self.graph.graph_top[ori]['nei'][node]['dist'] for node in self.graph.graph_top[ori]['nei']])

        # print(ori_nei)
        dest_nei_id = np.array([node for node in self.graph.graph_top[dest]['nei']])
        dest_nei_dist = np.array(
            [self.graph.graph_top[dest]['nei'][node]['dist'] for node in self.graph.graph_top[dest]['nei']])

        # passenger will walk to a node within 400 meters
        walk_dist = 400
        if ori not in self.near_nei:
            self.near_nei[ori] = ori_nei_id[np.where(ori_nei_dist <= walk_dist)[0]]
        # print(self.near_nei[ori])

        if len(self.near_nei[ori]) != 0:
            # pick the smallest queue length node
            ori_walk_nei_len = np.array([
                self.info['p_queue'][node]['taxi'][self.info['time'] - 1] - self.info['v_queue'][node]['taxi'][
                    self.info['time'] - 1]
                for node in self.near_nei[ori]])
            # print(self.info)
            # print(ori_walk_nei_len)
            ori_trans = self.near_nei[ori][
                np.where(ori_walk_nei_len == np.amin(ori_walk_nei_len))[0][0]
            ]
        else:
            # no nearby node within 400 meters, then go to the nearest one
            ori_trans = ori_nei_id[
                np.where(ori_nei_dist == np.amin(ori_nei_dist[
                                                     ori_nei_dist != np.amin(ori_nei_dist)
                                                     ]))[0][0]
            ]
        # the destination will always be the nearest node
        dest_trans = dest_nei_id[
            np.where(dest_nei_dist == np.amin(dest_nei_dist[
                                                  dest_nei_dist != np.amin(dest_nei_dist)
                                                  ]))[0][0]
        ]

        path.update({ori: {'dest': ori_trans, 'info': self.pathinfo_generator(ori=ori, dest=ori_trans, method='walk')}})
        if ori_trans != dest_trans:
            # walk to destination
            path.update({ori_trans: {'dest': dest_trans,
                                     'info': self.pathinfo_generator(ori=ori_trans, dest=dest_trans, method='taxi')}})

        path.update(
            {dest_trans: {'dest': dest, 'info': self.pathinfo_generator(ori=dest_trans, dest=dest, method='walk')}})

        # print(path)
        return path

    def bus_simplex(self, ori, dest):
        stops = ori
        path = {}
        # by scooter
        if dest in self.graph.graph_top[ori]['nei']:
            # path.append(self.get_edge(ori, dest))
            # print('dont need transfer')
            path.update({ori: {'dest': dest, 'info': self.pathinfo_generator(ori=ori, dest=dest, method='simplex')}})
            # print(path)
        else:
            # find nearest bus stop
            if 'bus' not in self.graph.graph_top[ori]['node'].mode:
                # print('find a bus stop')
                for busstop in self.graph.graph_top[ori]['nei']:
                    # print(self.graph_top[busstop]['mode'])
                    if 'bus' in self.graph.graph_top[busstop]['node'].mode:
                        # print(busstop)
                        # path.append(self.get_edge(ori, busstop))
                        path.update({ori: {'dest': busstop,
                                           'info': self.pathinfo_generator(ori=ori, dest=busstop, method='simplex')}})
                        stops = busstop

            # find transfer bus stop
            if 'bus' in self.graph.graph_top[dest]['node'].mode:
                # path.append(self.get_edge(stops, dest))
                # print('ready to get off')
                path.update(
                    {stops: {'dest': dest, 'info': self.pathinfo_generator(ori=stops, dest=dest, method='simplex')}})
            else:
                # travel by bus
                for busstop in self.graph.graph_top[dest]['nei']:
                    if 'bus' in self.graph.graph_top[busstop]['node'].mode:
                        # print('ready to transfer')
                        # path.append(self.get_edge(stops, busstop))
                        # path.append(self.get_edge(busstop, dest))
                        path.update({stops: {'dest': busstop, 'info': self.pathinfo_generator(ori=stops, dest=busstop,
                                                                                              method='simplex')}})
                        path.update({busstop: {'dest': dest, 'info': self.pathinfo_generator(ori=busstop, dest=dest,
                                                                                             method='simplex')}})
        return path

    def pathinfo_generator(self, ori, dest, method):
        edge = self.graph.get_edge(ori, dest)
        info = edge[2]
        mode = info['mode'].split(',')
        # print(mode)
        if len(mode) == 1:
            return info
        else:
            if method == 'simplex':
                return {'mode': mode[0], 'dist': info['dist']}
            elif method == 'walk':
                return {'mode': 'walk', 'dist': info['dist']} if ('walk' in mode) else {'mode': mode[0],
                                                                                        'dist': info['dist']}
            elif method == 'taxi':
                return {'mode': 'taxi', 'dist': info['dist']} if ('taxi' in mode) else {'mode': mode[0],
                                                                                        'dist': info['dist']}
            else:
                return {'mode': method, 'dist': info['dist']} if (method in mode) else {'mode': mode[0],
                                                                                        'dist': info['dist']}

    def save_path(self, ori, dest, method, path):
        # print(path)
        if not path:
            # print(path)
            return
        if ori not in self.path[method]:
            self.path[method][ori] = {}
            self.path[method][ori][dest] = path
            # print(path)
        elif dest not in self.path[method][ori]:
            self.path[method][ori][dest] = path
            # print(path)
        else:
            # print(path)
            return

    # the follows are algorithms
    # quick sort
    def partition(self, array, start, end):
        pivot = array[start]
        low = start + 1
        high = end

        while True:
            while low <= high and array[high] >= pivot:
                high = high - 1
            while low <= high and array[low] <= pivot:
                low = low + 1
            if low <= high:
                array[low], array[high] = array[high], array[low]
            else:
                break

        array[start], array[high] = array[high], array[start]

        return high

    def quick_sort(self, array, start, end):
        if start >= end:
            return

        p = self.partition(array, start, end)
        self.quick_sort(array, start, p - 1)
        self.quick_sort(array, p + 1, end)