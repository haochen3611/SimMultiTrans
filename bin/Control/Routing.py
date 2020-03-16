#!/usr/bin/env python
# -*- coding: utf-8 -*-
# from Node import Node

import numpy as np

import random

class Routing(object):
    def __init__(self, graph, vehicle_attri):
        self.graph = graph
        self.path = {
            'bus_simplex': {},
            'simplex': {},
            'bus_walk_simplex': {}
        }
        self.vehicle_attri = vehicle_attri

    def get_methods(self):
        return list(self.path.keys())

    def get_path(self, ori, dest, method='simplex'):
        if (ori not in self.graph.get_topology() and dest not in self.graph.get_topology()):
            # print('invalid path')
            return {}

        if (ori in self.path[method]) and (dest in self.path[method][ori]):
            # print('path exists')
            return self.path[method][ori][dest]
        else:
            methodset = {
                'bus_simplex': self.bus_simplex,
                'simplex': self.simplex,
                'bus_walk_simplex': self.bus_walk_simplex
            }
            # route = Routing(self, ori, dest)
            # print('routing: ', ori, dest)
            path = methodset[method](ori, dest)
            # print(path)
            self.save_path(ori, dest, method, path)
            return path

    def simplex(self, ori, dest):
        if ( dest in self.graph.get_topology()[ori]['nei'] ):
            return {ori: {'dest': dest, 'info': self.pathinfo_generator(ori=ori, dest=dest, method='simplex')}}
        else:
            return {}

    def bus_walk_simplex(self, ori, dest):
        # stops = ori
        path = {}
        
        # check avialability
        ori_nei_id = [ node for node in self.graph.get_topology()[ori]['nei'] ]
        ori_nei_dist = np.array( [ self.graph.get_topology()[ori]['nei'][node]['dist'] for node in self.graph.get_topology()[ori]['nei'] ] )
        
        # print(ori_nei)
        dest_nei_id = [ node for node in self.graph.get_topology()[dest]['nei'] ]
        dest_nei_dist = np.array( [ self.graph.get_topology()[dest]['nei'][node]['dist'] for node in self.graph.get_topology()[dest]['nei'] ] )

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

        ori_trans_modes = self.graph.get_topology()[ori_trans]['mode'].split(',')
        dest_trans_modes = self.graph.get_topology()[dest_trans]['mode'].split(',')
        mode = list(set(ori_trans_modes).intersection(dest_trans_modes))
        # no bus available, take taxi
        # also add some randomness
        if (len(mode) > 2 and random.choice([True, False])):
            path.update({ori: {'dest': ori_trans, 'info': self.pathinfo_generator(ori=ori, dest=ori_trans, method='walk')}})
            path.update({ori_trans: {'dest': dest_trans, 'info': self.pathinfo_generator(ori=ori_trans, dest=dest_trans, method=mode[0])}})
            path.update({dest_trans: {'dest': dest, 'info': self.pathinfo_generator(ori=dest_trans, dest=dest, method='walk')}})
            # return path
            # print(ori, dest)
            # print('a', path)
        # bus available
        else:
            path.update({ori: {'dest': ori_trans, 'info': self.pathinfo_generator(ori=ori, dest=ori_trans, method='walk')}})
            path.update({ori_trans: {'dest': dest_trans, 'info': self.pathinfo_generator(ori=ori_trans, dest=dest_trans, method='taxi')}})
            path.update({dest_trans: {'dest': dest, 'info': self.pathinfo_generator(ori=dest_trans, dest=dest, method='walk')}})
            # print(ori, dest)
            # print('b', path)
        return path

    def bus_simplex(self, ori, dest):
        stops = ori
        path = {}
        # by scooter
        if ( dest in self.graph.get_topology()[ori]['nei'] ):
            # path.append(self.get_edge(ori, dest))
            # print('dont need transfer')
            path.update({ori: {'dest': dest, 'info': self.pathinfo_generator(ori=ori, dest=dest, method='simplex')}})
            # print(path)
        else:
            # find nearest bus stop
            if ( 'bus' not in self.graph.get_topology()[ori]['node'].get_mode() ):
                # print('find a bus stop')
                for busstop in self.graph.get_topology()[ori]['nei']:
                    # print(self.graph_top[busstop]['mode'])
                    if ( 'bus' in self.graph.get_topology()[busstop]['node'].get_mode() ):
                        # print(busstop)
                        # path.append(self.get_edge(ori, busstop))
                        path.update({ori: {'dest': busstop, 'info': self.pathinfo_generator(ori=ori, dest=busstop, method='simplex')}})
                        stops = busstop
            
            # find transfer bus stop
            if ( 'bus' in self.graph.get_topology()[dest]['node'].get_mode() ):
                # path.append(self.get_edge(stops, dest))
                # print('ready to get off')
                path.update({stops: {'dest': dest, 'info': self.pathinfo_generator(ori=stops, dest=dest, method='simplex')}})
            else:
                # travel by bus
                for busstop in self.graph.get_topology()[dest]['nei']:
                    if ( 'bus' in self.graph.get_topology()[busstop]['node'].get_mode() ):
                        # print('ready to transfer')
                        # path.append(self.get_edge(stops, busstop))
                        # path.append(self.get_edge(busstop, dest))
                        path.update({stops: {'dest': busstop, 'info': self.pathinfo_generator(ori=stops, dest=busstop, method='simplex')}})
                        path.update({busstop: {'dest': dest, 'info': self.pathinfo_generator(ori=busstop, dest=dest, method='simplex')}})
        return path

    def pathinfo_generator(self, ori, dest, method):
        edge = self.graph.get_edge(ori, dest)
        info = edge[2]
        mode = info['mode'].split(',')
        if ( len(mode) == 1 ):
            return info
        else:
            if (method == 'simplex'):
                return {'mode': mode[0], 'dist': info['dist']}
            elif (method == 'walk'):
                return {'mode': 'walk', 'dist': info['dist']} if ('walk' in mode) else {'mode': mode[0], 'dist': info['dist']}
            elif (method == 'taxi'):
                return {'mode': 'taxi', 'dist': info['dist']} if ('taxi' in mode) else {'mode': mode[0], 'dist': info['dist']}
            else:
                return {'mode': method, 'dist': info['dist']} if (method in mode) else {'mode': mode[0], 'dist': info['dist']}

    def save_path(self, ori, dest, method, path):
        if (ori not in self.path[method]):
            self.path[method][ori] = {}
            self.path[method][ori][dest] = path
        elif (dest not in self.path[method]):
            self.path[method][ori][dest] = path
        else:
            return


    # the follows are algorithms
    # quick sort
    def partition(self, array, start, end):
        pivot = array[start]
        low = start + 1
        high = end

        while True:
            while (low <= high and array[high] >= pivot):
                high = high - 1
            while (low <= high and array[low] <= pivot):
                low = low + 1
            if (low <= high):
                array[low], array[high] = array[high], array[low]
            else:
                break

        array[start], array[high] = array[high], array[start]

        return high
    
    def quick_sort(self, array, start, end):
        if (start >= end):
            return

        p = partition(array, start, end)
        quick_sort(array, start, p-1)
        quick_sort(array, p+1, end)