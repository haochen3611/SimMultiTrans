#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class Graph(object):
    def __init__(self):
        self.graph_dic = {
            'A': [('A1',1),('A2',1),('A3',1), ('B',2),('C',2)],
            'A1': [('A',1), ('A2',1),('A3',1)],
            'A2': [('A',1), ('A1',1),('A3',1)],
            'A3': [('A',1), ('A2',1),('A1',1)],
            'B': [('B1',1),('B2',1),('B3',1), ('A',2),('C',2)],
            'B1': [('B',1), ('B2',1),('B3',1)],
            'B2': [('B',1), ('B1',1),('B3',1)],
            'B3': [('B',1) ,('B2',1),('B1',1)],
            'C': [('C1',1),('C2',1),('C3',1), ('B',2),('A',2)],
            'C1': [('C',1), ('C1',1),('C2',1)],
            'C2': [('C',1), ('C1',1),('C3',1)],
            'C3': [('C',1), ('C2',1),('C1',1)]
        }

        self.garph_path = {}

    def get_allnodes(self):
        return list(self.graph_dic.keys())

    def get_edge(self, ori, dest):
        if (ori in self.graph_dic) and (dest in self.graph_dic):
            for e in self.graph_dic[ori]:
                if e[0] == dest:
                    return (ori, e)


    def get_path(self, ori, dest): 
        if (ori in self.garph_path) and (dest not in self.garph_path[ori]):
            return self.garph_path[ori][dest]
        else:
            path = []
            dest_set = [ d[0] for d in self.graph_dic[ori] ]
            if (dest in dest_set):
                path.append(self.get_edge(ori, dest))
            else:
                if (ori != ori[0]):
                    path.append(self.get_edge(ori, ori[0]))
                for d in self.graph_dic[ori[0]]:
                    if (d[0] == dest[0]):
                        path.append(self.get_edge(ori[0], dest[0]))
                
                if (len(dest) != 1):
                    path.append(self.get_edge(dest[0], dest))
            return path
        
    def save_path(self, ori, dest, path):
        if (ori in self.garph_path) and (dest not in self.garph_path[ori]):
            return
        else:
            self.garph_path[ori][dest] = {path}


    