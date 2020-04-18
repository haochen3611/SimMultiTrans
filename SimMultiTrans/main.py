#!/usr/bin/env python
# -*- coding: utf-8 -*-

from SimMultiTrans.bin.Control import *
from SimMultiTrans.bin.Network import *
from SimMultiTrans.bin import *


def main():
    lazyset = [1, 10, 20, 25, 40]
    rangeset = [10, 20, 50]
    policyset = [
        'Simplified_MaxWeight', 'Proportional'
    ]

    '''
    for l in lazyset:
        for r in rangeset:

    for p_name in policyset:
        for r in rangeset:
            for l in lazyset:
    '''

    l = 25
    r = 10
    p_name = 'None'
    # r_name = 'taxi_walk_simplex'
    r_name = 'simplex'

    # create graph
    g = Graph()
    # g.import_graph(file_name='conf/city_ct.json')
    g.import_graph(file_name='conf/city_1.json')

    # setup simulator
    simu = Simulator(graph=g)
    simu.import_arrival_rate(unit=(1, 'sec'))
    simu.import_vehicle_attribute(file_name='conf/vehicle_1.json')
    # simu.import_vehicle_attribute(file_name='conf/vehicle_nyc.json')
    # simu.import_arrival_rate(unit=(1,'sec'))
    # simu.import_vehicle_attribute(file_name='conf/vehicle_ct.json')
    simu.set_multiprocessing(False)

    # simu.set_running_time(starttime='08:00:00', timehorizon=1, unit='hour')
    simu.set_running_time(starttime='08:00:00', timehorizon=1.84, unit='hour')

    simu.routing.set_routing_method(r_name)
    simu.rebalance.set_parameters(lazy=l, vrange=r)
    simu.rebalance.set_policy(p_name)

    simu.initialize(seed=0)
    simu.run()
    simu.save_result(path_name=f'results/{p_name}__l={simu.rebalance.lazy}r={simu.rebalance.range}')

    '''
    simu.initialize(seed=0)

    for l in lazyset:
        simu.rebalance.set_parameters(lazy=l, vrange=r)
        simu.run()
        simu.save_result(path_name=f'results/{p_name}__l={simu.rebalance.lazy}r={simu.rebalance.range}')
    '''

    # simu.plot.combination_queue_animation(mode='taxi', frames=100, autoplay=True)

    # plot = Plot(simu.graph, simu.time_horizon, simu.start_time)
    # plot.set_plot_theme('plotly_dark')
    # plot.plot_topology()
    '''
    plot = Plot(simu.graph, simu.time_horizon, simu.start_time)
    plot.plot_metrics_animation('taxi', ['MaxWeight__l=1r=50', 'Simplified_CostSensitive__l=10r=50',
        'CostSensitive__l=25r=10', 'Simplified_MaxWeight__l=20r=50', 'Proportional__l=20r=50', 'None__l=20r=20'
    ])


    res = [
        f'{p_name}__l={l}r={r}' 
        for l in lazyset
    ]
    plot = Plot(simu.graph, simu.time_horizon, simu.start_time)
    plot.plot_metrics_animation('taxi', res)
    '''
    plot = Plot(simu.graph, simu.time_horizon, simu.start_time)
    plot.import_results(path_name=f'results/{p_name}__l={simu.rebalance.lazy}r={simu.rebalance.range}')
    # plot.combination_queue_animation(mode='taxi', frames=100, autoplay=True)
    plot.plot_passenger_queuelen_time(mode='taxi')
    plot.plot_passenger_waittime(mode='taxi')
    # plot.plot_metrics(mode='taxi')


if __name__ == "__main__":
    main()