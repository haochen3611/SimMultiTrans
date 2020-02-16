#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Graph import Graph
from Passenger import Passenger
from Vehicle import Vehicle

from Simulator import Simulator

import numpy as np
import random


def main():
    # create graph
    g = Graph()
    g.import_graph(file_name='city.json')
    # g.plot_topology()

    # setup simulator
    simu = Simulator(graph=g, time_horizon=10000)
    simu.start()
    # simu.plot_passenger_queuelen(10000-1)
    simu.animation(100)

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

        
        


if __name__ == "__main__":
    main()