#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import json

from Node import Node

class Graph(object):
    def __init__(self):
        # generate graph
        self.graph_top = {}

        # path ram
        self.garph_path = {}

    def set_graph(self, graph):
        self.graph_top = graph

    def import_graph(self, file_name):
        with open('{}'.format(file_name)) as file_data:
            self.graph_top = json.load(file_data)
 
    def generate_node(self, nid):
        n = Node( nid, (self.graph_top[nid]['locx'], self.graph_top[nid]['locy']), self.graph_top[nid]['mode'].split(',') )
        # print(n.get_id())
        self.graph_top[nid].update({'node': n})
            

    def generate_nodes(self):
        for node in self.graph_top.keys():
            # print(node, (self.graph_top[node]['locx'], self.graph_top[node]['locy']), self.graph_top[node]['mode'].split(','))
            n = Node( node, (self.graph_top[node]['locx'], self.graph_top[node]['locy']), self.graph_top[node]['mode'].split(',') )
            # print(n.get_id())
            self.graph_top[node].update({'node': n})

    def export_nodes(self, file_name):
        with open('{}'.format(file_name), 'w') as file_data:
            json.dump(self.graph_top, file_data)

    def add_node(self, nid, locx, locy, mode):
        if ( nid not in self.graph_top ):
            self.graph_top[nid] = {}
            self.graph_top[nid]['locx'] = locx
            self.graph_top[nid]['locy'] = locy
            self.graph_top[nid]['mode'] = mode
            self.graph_top[nid]['nei'] = {}
            

    def add_edge(self, ori, dest, mode, dist):
        if ( ori in self.graph_top and dest in self.graph_top ):
            self.graph_top[ori]['nei'][dest] = {'mode': mode, 'dist': dist}
            

    def get_allnodes(self):
        return list(self.graph_top.keys())

    def get_edge(self, ori, dest):
        if (ori in self.graph_top) and (dest in self.graph_top):
            if dest in self.graph_top[ori]['nei'].keys():
                return (ori, dest, self.graph_top[ori]['nei'][dest])

    def get_topology(self):
        return self.graph_top

    def node_exists(self, node):
        return (node in self.graph_top)

    def edge_exists(self, ori, dest):
        if ( self.node_exists(ori) and self.node_exists(dest) ):
            return ( dest in self.graph_top[ori]['nei'] )
        else:
            return False

    def get_node_location(self, node):
        return (self.graph_top[node]['locx'], self.graph_top[node]['locy'])
    
    def get_L1dist(self, ori, dest):
        if ( self.node_exists(ori) and self.node_exists(dest) ):
            return np.abs(self.graph_top[ori]['locx']-self.graph_top[dest]['locx']) + np.abs(self.graph_top[ori]['locy']-self.graph_top[dest]['locy'])
        else:
            return 0

    def get_path(self, ori, dest): 
        if (ori not in self.graph_top and dest not in self.graph_top):
            return []

        if (ori in self.garph_path) and (dest in self.garph_path[ori]):
            # print('path exists')
            return self.garph_path[ori][dest]
        else:
            path = []
            stops = ori
            # by scooter
            if (dest in self.graph_top[ori]['nei']):
                path.append(self.get_edge(ori, dest))
            else:
                # find nearest bus stop
                if ( 'bus' not in self.graph_top[ori]['node'].get_mode() ):
                    # print('find a bus stop')
                    for busstop in self.graph_top[ori]['nei']:
                        # print(self.graph_top[busstop]['mode'])
                        if ( 'bus' in self.graph_top[busstop]['node'].get_mode() ):
                            # print(busstop)
                            path.append(self.get_edge(ori, busstop))
                            stops = busstop
                
                # find transfer bus stop
                if ( 'bus' in self.graph_top[dest]['node'].get_mode() ):
                    path.append(self.get_edge(stops, dest))
                else:
                    # travel by bus
                    for busstop in self.graph_top[dest]['nei']:
                        if ( 'bus' in self.graph_top[busstop]['node'].get_mode() ):
                            path.append(self.get_edge(stops, busstop))
                            path.append(self.get_edge(busstop,dest))
            return path
        
    def save_path(self, ori, dest, path):
        if (ori in self.garph_path) and (dest not in self.garph_path[ori]):
            return
        else:
            self.garph_path[ori][dest] = {path}

