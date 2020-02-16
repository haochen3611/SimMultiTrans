#!/usr/bin/env python
# -*- coding: utf-8 -*-
from Node import Node

import numpy as np
import matplotlib.pyplot as plt

import json


class Graph(object):
    def __init__(self):
        self.graph_top = {}

        # all edges
        self.edges_set = []

        # path ram
        self.garph_path = {}

    def set_graph(self, graph):
        self.graph_top = graph

    def get_graph_dic(self):
        return self.graph_top

    def import_graph(self, file_name):
        with open('{}'.format(file_name)) as file_data:
            self.graph_top = json.load(file_data)
 
    def generate_node(self, nid):
        n = Node( nid, self.graph_top )
        # print(n.get_id())
        self.graph_top[nid].update({'node': n})
            

    def generate_nodes(self):
        for node in self.graph_top.keys():
            # print(node, (self.graph_top[node]['locx'], self.graph_top[node]['locy']), self.graph_top[node]['mode'].split(','))
            n = Node( node, self.graph_top )
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
        '''return (ori, dest) edges'''
        if (ori in self.graph_top) and (dest in self.graph_top):
            if dest in self.graph_top[ori]['nei'].keys():
                return (ori, dest, self.graph_top[ori]['nei'][dest])

    def get_all_edges(self):
        '''return all edges'''
        if ( not self.edges_set ):
            for ori in self.graph_top:
                self.edges_set.append( [ (ori ,dest) for dest in self.graph_top[ori]['nei'] ] )
        return self.edges_set
        

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


    def randomize_graph(self, seed, msize, modeset, max_localnodes, map_scale):
        np.random.seed(seed)
        M = np.random.randint(2, size=(msize, msize))
        self.graph_top = {}
        self.garph_path = {}
        
        transfer_mode = ','.join(modeset)

        loc_set = np.random.randint(low=0, high=map_scale*msize, size=(msize, 2))
        
        # generage transfer nodes and edges
        for ori in range(msize):
            self.add_node(nid=chr(65+ori), locx=loc_set[ori][0], locy=loc_set[ori][1], mode=transfer_mode)
            # g.generate_node(nid='{}'.format(ori))

            for dest in self.get_allnodes():
                if (ori == dest):
                    break
                else:
                    dist = self.get_L1dist(ori, dest)
                    # symmetric edge
                    self.add_edge(ori=ori, dest=dest, mode=modeset[0], dist=dist)
                    self.add_edge(ori=dest, dest=ori, mode=modeset[0], dist=dist)
        
        print(self.get_allnodes())
        # generate local nodes
        for t_node in self.get_allnodes():
            M = int(np.random.randint(max_localnodes, size=1))
            (x, y) = self.get_node_location(t_node) 

            for l_node in range(M):
                x = x + round(map_scale/msize * np.random.normal(1) ,2)
                y = y + round(map_scale/msize * np.random.normal(1) ,2)
                nid = t_node+chr(49+l_node)
                self.add_node(nid=nid, locx=x, locy=y, mode=modeset[1])
                # g.generate_node(nid='{}'.format(l_node))
                
                dist = self.get_L1dist(t_node, nid)
                self.add_edge(ori=t_node, dest=nid, mode=modeset[1], dist=dist)
                self.add_edge(ori=nid, dest=t_node, mode=modeset[1], dist=dist)


    def plot_topology(self):
        fig, ax = plt.subplots()
        
        x = [ self.get_node_location(node)[0] for node in self.graph_top ]
        y = [ self.get_node_location(node)[1] for node in self.graph_top ]

        '''
        alledges = self.get_all_edges()
        # print(alledges)
        loc = np.zeros(shape=(2,2))

        for odlist in alledges:
            for odpair in odlist:
                loc[:,0] = np.array( [self.graph_top[odpair[0]]['locx'], self.graph_top[odpair[0]]['locy']])
                loc[:,1] = np.array( [self.graph_top[odpair[1]]['locx'], self.graph_top[odpair[1]]['locy']])
                ax.plot(loc[0,:], loc[1,:], c='grey', alpha=0.5, ls='--', lw=2, zorder=1)
        '''
        fig, ax = self.plot_alledges(x,y)

        # color = np.random.randint(1, 100, size=len(self.get_allnodes()))
        color = [ 'steelblue' if (',' in self.graph_top[node]['mode']) else 'skyblue' for node in self.graph_top ]
        scale = [ 300 if (',' in self.graph_top[node]['mode']) else 100 for node in self.graph_top ]

        ax.scatter(x, y, c=color, s=scale, label=color, alpha=0.8, edgecolors='none', zorder=2)

        # ax.legend()
        plt.grid(True)
        # plt.legend(loc='lower right', framealpha=1)
        plt.xlabel('Latitude')
        plt.ylabel('Longitude')
        plt.title('City Topology')

        plt.savefig('City_Topology.png', dpi=600)
        # plt.show()

    def plot_alledges(self, x, y):
        fig, ax = plt.subplots()

        alledges = self.get_all_edges()
        # print(alledges)
        loc = np.zeros(shape=(2,2))

        for odlist in alledges:
            for odpair in odlist:
                loc[:,0] = np.array( [self.graph_top[odpair[0]]['locx'], self.graph_top[odpair[0]]['locy']])
                loc[:,1] = np.array( [self.graph_top[odpair[1]]['locx'], self.graph_top[odpair[1]]['locy']])
                ax.plot(loc[0,:], loc[1,:], c='grey', alpha=0.5, ls='--', lw=2, zorder=1)

        # plt.show()
        return fig, ax