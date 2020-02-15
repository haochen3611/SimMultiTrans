#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Graph import Graph
from Passenger import Passenger
from Vehicle import Vehicle
from Converter import Converter

import numpy as np
import random


def main():
    g = Graph()
    g.import_graph(file_name='city.json')
    g.generate_nodes()
    '''
    g.add_node(nid='a', locx=1, locy=2, mode='scooter')
    g.add_node(nid='b', locx=2, locy=2, mode='scooter,bus')
    g.add_node(nid='c', locx=1, locy=2, mode='bus,scooter')
    g.add_edge(ori='a',dest='b', mode='scooter',dist='100')
    g.add_edge(ori='b',dest='c', mode='bus',dist='1000')
    
    converter = Converter(1)
    g.set_graph( converter.randon_graph(msize=6, modeset=['bus','scooter']) )
    g.generate_nodes()
    print(g.get_topology())
    # g.plot_topology()
    '''

    for time in range(50):
        print('time=', time)
        for node in g.get_allnodes():
            print('node=', node)
            g.graph_top[node]['node'].syn_time(time)
            g.graph_top[node]['node'].new_passenger_generator(g)

    # p = Passenger(pid=1, ori='a', dest='c', arr_time=0)
    # print(p.get_schdule(graph=g))
        
        


if __name__ == "__main__":
    main()