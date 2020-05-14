#!/usr/bin/env python
# -*- coding: utf-8 -*-

from SimMultiTrans.bin.Network import *
from SimMultiTrans.bin import *
import numpy as np
import time


def main():

    lazy = 25
    range_ = 10
    p_name = 'None'
    r_name = 'simplex'
    step_len = 600
    time_hor = 1
    # create graph
    g = Graph()
    g.import_graph(file_name='conf/city_nyc.json')

    # setup simulator
    simu = Simulator(graph=g)
    simu.import_arrival_rate(unit=(1, 'sec'))
    simu.import_vehicle_attribute(file_name='conf/vehicle_nyc.json')
    simu.set_running_time(start_time='08:00:00', time_horizon=time_hor, unit='hour')
    # simu.routing.set_routing_method(r_name)
    # simu.rebalance.set_parameters(lazy=lazy, v_range=range_)
    # simu.rebalance.set_policy(p_name)

    simu.initialize(seed=0)
    action = np.random.random((len(g.get_all_nodes()), len(g.get_all_nodes())))
    sim_action = dict()
    for idx, node in enumerate(g.get_all_nodes()):
        sim_action[node] = action[idx, :]/np.sum(action[idx, :])
    c_time = 0
    start_time = time.time()
    for idx in range(time_hor*3600//step_len):
        print(c_time)
        simu.step(action=sim_action,
                  step_length=step_len,
                  curr_time=c_time)
        c_time += step_len
    simu.finishing_touch(start_time)
    # simu.save_result(path_name=f'results/test.json')

    # plot = Plot(simu.graph, simu.time_horizon, simu.start_time)
    # plot.import_results(path_name=f'results/{p_name}__l={simu.rebalance.lazy}r={simu.rebalance.range}')
    # plot.combination_queue_animation(mode='taxi', frames=100, autoplay=True)
    # plot.plot_passenger_queuelen_time(mode='taxi')
    # plot.plot_passenger_waittime(mode='taxi')
    # plot.plot_metrics(mode='taxi')


if __name__ == "__main__":
    import cProfile, pstats, io
    from pstats import SortKey
    pr = cProfile.Profile()
    pr.enable()
    main()
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(SortKey.CUMULATIVE)
    ps.print_stats()
    print(s.getvalue())
