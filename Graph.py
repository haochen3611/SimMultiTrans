#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import json

import logging

from Node import Node

class Graph(object):
    def __init__(self, file_name):
        # generate graph
        self.graph_dic = self.set_nodes_file(file_name)

        # path ram
        self.garph_path = {}


    def set_nodes_file(self, file_name):
        with open('{}'.format(file_name)) as file_data:
            return json.load(file_data)
 
    def generate_nodes(self):
        for node in self.graph_dic.keys():
            # print(node, (self.graph_dic[node]['locx'], self.graph_dic[node]['locy']), self.graph_dic[node]['mode'].split(','))
            n = Node( node, (self.graph_dic[node]['locx'], self.graph_dic[node]['locy']), self.graph_dic[node]['mode'].split(','))
            # print(n.get_id())
            self.graph_dic[node].update({'node': n})

    def save_nodes_file(self, file_name):
        with open('{}'.format(file_name), 'w') as file_data:
            json.dump(self.graph_dic, file_data)


    def get_allnodes(self):
        return list(self.graph_dic.keys())

    def get_edge(self, ori, dest):
        if (ori in self.graph_dic) and (dest in self.graph_dic):
            if dest in self.graph_dic[ori]['nei'].keys():
                return (ori, dest, self.graph_dic[ori]['nei'][dest])
                    

    def get_path(self, ori, dest): 
        if (ori in self.garph_path) and (dest in self.garph_path[ori]):
            # print('path exists')
            return self.garph_path[ori][dest]
        else:
            path = []
            stops = ori
            # by scooter
            if (dest in self.graph_dic[ori]['nei']):
                path.append(self.get_edge(ori, dest))
            else:
                # find nearest bus stop
                if ( 'bus' not in self.graph_dic[ori]['mode'].split(',') ):
                    # print('find a bus stop')
                    for busstop in self.graph_dic[ori]['nei']:
                        # print(self.graph_dic[busstop]['mode'])
                        if ('bus' in self.graph_dic[busstop]['mode'].split(',')):
                            print(busstop)
                            path.append(self.get_edge(ori, busstop))
                            stops = busstop
                
                # find transfer bus stop
                if ( 'bus' in self.graph_dic[dest]['mode'].split(',') ):
                    path.append(self.get_edge(stops, dest))
                else:
                    # travel by bus
                    for busstop in self.graph_dic[dest]['nei']:
                        if ('bus' in self.graph_dic[busstop]['mode'].split(',')):
                            path.append(self.get_edge(stops, busstop))
                            path.append(self.get_edge(busstop,dest))
            return path
        
    def save_path(self, ori, dest, path):
        if (ori in self.garph_path) and (dest not in self.garph_path[ori]):
            return
        else:
            self.garph_path[ori][dest] = {path}


    