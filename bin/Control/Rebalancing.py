#!/usr/bin/env python
# -*- coding: utf-8 -*-
# from Node import Node

import numpy as np
from scipy.optimize import linprog

class Rebalancing(object):
    def __init__(self, graph, vehicle_attri):
        self.graph = graph
        self.vehicle_attri = vehicle_attri

    def MaxWeight(self, node, queue, server):
        if (np.sum(queue) == 0):
            return np.zeros(len(queue))

        if (len(queue) == len(server)):
            # A_eq = np.ndarray(np.ones(shape=(1, len(queue))))
            # b_eq = np.sum(server)
            # print('q', queue)
            # print('s', server)
            # print(A_eq, b_eq)
            # result = linprog(c=queue, A_eq=A_eq, b_eq=b_eq, bounds=[0, np.inf], method='simplex')
            A_ub = np.eye(len(queue))
            dist_list = np.array( [ self.graph.get_topology()[node]['nei'][dest]['dist']
                for dest in self.graph.get_topology()[node]['nei'] ] )
            k = 4
            k_near_list = dist_list.argsort()[:k]
            b_ub = np.zeros(shape=(len(queue), 1))
            # need review!!
            b_ub[k_near_list] = sum(server)/2

            A_eq = np.ones(shape=(1, len(queue)))
            b_eq = sum(server)
            c = -np.array(queue)
            result = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[0, sum(server)], method='simplex')
            # print(result.x)
            return result.x
        return None

    def Dispatch_active(self, node, mode, queue_p, queue_v):
        if (self.vehicle_attri[mode]['reb'] == 'active'):
            opt_queue_v = self.MaxWeight(node=node, queue=queue_p, server=queue_v)
            # normalize
            sum_queue = np.sum(opt_queue_v)
            # print(sum_queue)

            return (opt_queue_v / sum_queue) if (sum_queue != 0) else np.zeros(len(opt_queue_v))
        else: 
            return None
        