#!/usr/bin/env python
# -*- coding: utf-8 -*-
from Node import Node

import numpy as np

class Routing(object):
    def __init__(self, graph, vehicle_attri):
        self.graph = graph
        self.path = {
            'bus_simplex': {},
            'simplex': {}
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
                'simplex': self.simplex
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
            elif (method == 'taxi'):
                return {'mode': 'taxi', 'dist': info['dist']} if ('taxi' in mode) else {'mode': mode[0], 'dist': info['dist']}
            else:
                return {'mode': mode[0], 'dist': info['dist']}

    def save_path(self, ori, dest, method, path):
        if (ori not in self.path[method]):
            self.path[method][ori] = {}
            self.path[method][ori][dest] = path
        elif (dest not in self.path[method]):
            self.path[method][ori][dest] = path
        else:
            return