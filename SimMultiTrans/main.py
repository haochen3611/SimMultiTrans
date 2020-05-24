#!/usr/bin/env python
# -*- coding: utf-8 -*-

from SimMultiTrans.bin.Network import *
from SimMultiTrans.bin import *
from SimMultiTrans import graph_file, vehicle_file
from SimMultiTrans.utils import RESULTS
import numpy as np
import time


def main():
    start_time = time.time()
    lazy = 25
    range_ = 10
    p_name = 'None'
    r_name = 'simplex'
    step_len = 600
    time_hor = 10
    # create graph
    g = Graph()
    g.import_graph(file_name=graph_file)

    # setup simulator
    simu = Simulator(graph=g)
    simu.import_arrival_rate(unit=(1, 'sec'))
    simu.import_vehicle_attribute(file_name=vehicle_file)
    simu.set_running_time(start_time='08:00:00', time_horizon=time_hor, unit='hour')
    # simu.routing.set_routing_method(r_name)
    # simu.rebalance.set_parameters(lazy=lazy, v_range=range_)
    # simu.rebalance.set_policy(p_name)

    simu.initialize(seed=0)
    # action = np.random.random((len(g.get_all_nodes()), len(g.get_all_nodes())))
    action = np.eye(len(g.get_all_nodes()))
    # action = np.zeros((len(g.get_all_nodes()), len(g.get_all_nodes())))
    sim_action = dict()
    for idx, node in enumerate(g.get_all_nodes()):
        sim_action[node] = action[idx, :]/np.sum(action[idx, :]) if np.sum(action[idx, :]) != 0 else 0
    c_time = 0
    for idx in range(time_hor*3600//step_len):
        print(c_time)
        simu.step(action=sim_action,
                  step_length=step_len,
                  curr_time=c_time)
        c_time += step_len
    simu.finishing_touch(start_time)
    simu.save_result(path_name=RESULTS, unique_name=False)
    print("Time used:", time.time() - start_time)

    plot = Plot(simu.graph, simu.time_horizon, simu.start_time)
    plot.import_results(path_name=RESULTS)
    plot.combination_queue_animation(mode='taxi', frames=100)
    plot.plot_passenger_queuelen_time(mode='taxi')
    plot.plot_passenger_waittime(mode='taxi')
    plot.plot_metrics(mode='taxi')


if __name__ == "__main__":
    # import cProfile, pstats, io
    # from pstats import SortKey
    # pr = cProfile.Profile()
    # pr.enable()
    main()
    # pr.disable()
    # s = io.StringIO()
    # ps = pstats.Stats(pr, stream=s).sort_stats(SortKey.CUMULATIVE)
    # ps.print_stats()
    # print(s.getvalue())
