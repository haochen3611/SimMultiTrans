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
    l = 0
    r = 20
    p_name = 'Simplified_MaxWeight'
    r_name = 'taxi_walk_simplex'
    # create graph
    g = Graph()
    g.import_graph(file_name='conf/city_ct.json')

    # setup simulator
    simu = Simulator(graph=g)
    # simu.import_arrival_rate(unit=(1,'sec'))
    # simu.import_vehicle_attribute(file_name='conf/vehicle_nyc.json')
    simu.import_arrival_rate(unit=(1,'sec'))
    simu.import_vehicle_attribute(file_name='conf/vehicle_ct.json')
    simu.set_multiprocessing(False)

    # simu.set_running_time(starttime='08:00:00', timehorizon=1, unit='hour')
    simu.set_running_time(starttime='08:00:00', timehorizon=2.5, unit='hour')

    
    simu.routing.set_routing_method(r_name)
    simu.rebalance.set_parameters(lazy=l, vrange=r)
    simu.rebalance.set_policy(p_name)
    
    '''
    simu.run()
    simu.save_result(path_name=f'results/{p_name}_l={simu.rebalance.lazy}r={simu.rebalance.range}')
    '''

    plot = Plot(simu.graph, simu.time_horizon, simu.start_time)
    plot.set_plot_theme('plotly_dark')
    plot.import_results(path_name=f'results/{p_name}_l={simu.rebalance.lazy}r={simu.rebalance.range}')
    # plot.combination_queue_animation(mode='taxi', frames=100, autoplay=True)
    # plot.plot_passenger_queuelen_time(mode='taxi')
    # plot.plot_passenger_waittime(mode='taxi')
    plot.plot_metrics(mode='taxi')

if __name__ == "__main__":
    main()