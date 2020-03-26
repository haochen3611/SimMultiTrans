#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
from Graph import Graph
from Passenger import Passenger
from Vehicle import Vehicle
from Simulator import Simulator
'''
from bin.Control import *
from bin.Network import *
from bin import *

def main():
    lazyset = [0.001, 1, 10, 100, 10000]
    rangeset = [2, 5, 10, 20, 40]

    '''
    for l in lazyset:
        for r in rangeset:
    '''
    l = 10
    r = 20
    # create graph
    g = Graph()
    # g.randomize_graph(seed=10, msize=10, modeset=['bus','scooter'], max_localnodes=5, mapscale=1000)
    # g.import_graph(file_name='city_nyc.json')
    g.import_graph(file_name='conf/city_ct.json')
    # g.plot_topology(method='plotly')

    # setup simulator
    simu = Simulator(graph=g)
    # simu.import_arrival_rate(unit=(1,'sec'))
    # simu.import_vehicle_attribute(file_name='vehicle_nyc.json')
    simu.import_arrival_rate(unit=(1,'sec'))
    simu.import_vehicle_attribute(file_name='conf/vehicle_ct.json')
    simu.set_multiprocessing(False)

    # simu.set_running_time(timehorizon=1, unit='hour')
    simu.set_running_time(starttime='08:00:00', timehorizon=2.5, unit='hour')

    simu.rebalance.lazy = l
    simu.rebalance.range = r
    # simu.plot_topology()
    simu.run()
    simu.save_result(path_name=f'results/l={simu.rebalance.lazy}r={simu.rebalance.range}')
    # simu.plot_passenger_queuelen(mode='scooter', time='06:05:00')
    # simu.passenger_queue_animation(mode='scooter', frames=30, autoplay=True)
    # simu.vehicle_queue_animation(mode='scooter', frames=30, autoplay=True)
    # simu.combination_queue_animation(mode='taxi', frames=100, autoplay=True)
    # simu.combination_queue_animation(mode='bus', frames=100, autoplay=True)
    # simu.passenger_queuelen_time(mode='taxi')
    # simu.passegner_waittime(mode='taxi')

    plot = Plot(simu.graph, simu.time_horizon, simu.start_time)
    plot.import_results(path_name=f'results/l={simu.rebalance.lazy}r={simu.rebalance.range}')
    plot.combination_queue_animation(mode='taxi', frames=100, autoplay=True)
    plot.plot_passenger_queuelen_time(mode='taxi')
    plot.plot_passenger_waittime(mode='taxi')

if __name__ == "__main__":
    main()