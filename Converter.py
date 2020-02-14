#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Graph import Graph

import numpy as np

class Converter(object):
    def __init__(self, seed):
        self.seed = seed

    def randon_graph(self, msize, modeset):
        '''create a graph with matrix M'''
        g = Graph()
        M = np.random.randint(2, size=(msize, msize))
        
        transfer_mode = ','.join(modeset)

        max_localnodes = 4
        
        loc_set = np.random.randint(low=0, high=20*msize, size=(msize, 2))
        
        # generage transfer nodes and edges
        for ori in range(msize):
            g.add_node(nid=chr(65+ori), locx=loc_set[ori][0], locy=loc_set[ori][1], mode=transfer_mode)
            # g.generate_node(nid='{}'.format(ori))

            for dest in g.get_allnodes():
                if (ori == dest):
                    break
                else:
                    dist = g.get_L1dist(ori, dest)
                    # symmetric edge
                    g.add_edge(ori=ori, dest=dest, mode=modeset[0], dist=dist)
                    g.add_edge(ori=dest, dest=ori, mode=modeset[0], dist=dist)
        
        print(g.get_allnodes())
        # generate local nodes
        for t_node in g.get_allnodes():
            M = int(np.random.randint(max_localnodes, size=1))
            (x, y) = g.get_node_location(t_node) 

            for l_node in range(M):
                x = x + np.random.normal(1)
                y = y + np.random.normal(1)
                nid = t_node+chr(49+l_node)
                g.add_node(nid=nid, locx=x, locy=y, mode=modeset[1])
                # g.generate_node(nid='{}'.format(l_node))
                
                dist = round(g.get_L1dist(t_node, nid),2)
                g.add_edge(ori=t_node, dest=nid, mode=modeset[1], dist=dist)
                g.add_edge(ori=nid, dest=t_node, mode=modeset[1], dist=dist)

        return g.get_topology()
