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
    # g.randomize_graph(seed=10, msize=10, modeset=['bus','scooter'], max_localnodes=5, mapscale=1000)
    g.import_graph(file_name='city.json')
    # g.plot_topology(method='plotly')

    # setup simulator
    simu = Simulator(graph=g)
    simu.import_arrival_rate(file_name='arr_rate.csv', unit='hour')
    simu.import_vehicle_attribute(file_name='vehicle.json')

    # simu.set_running_time(timehorizon=1, unit='hour')
    simu.set_running_time(timehorizon=4, unit='hour')
    simu.start()
    # simu.plot_passenger_queuelen(10000-1)
    # simu.passenger_queue_animation(mode='scooter', frames=30, autoplay=True, method='plotly')
    # simu.vehicle_queue_animation(mode='scooter', frames=100, autoplay=True, method='plotly')
    simu.combination_queue_animation(mode='scooter', frames=100, autoplay=True, method='plotly')
    # simu.combination_queue_animation(mode='bus', frames=100, autosave=True)

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