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
        if (len(queue) == len(server)):
            # A_eq = np.ndarray(np.ones(shape=(1, len(queue))))
            # b_eq = np.sum(server)
            # print('q', queue)
            # print('s', server)
            # print(A_eq, b_eq)
            # result = linprog(c=queue, A_eq=A_eq, b_eq=b_eq, bounds=[0, np.inf], method='simplex')
            result = linprog(c=queue, bounds=[0, np.inf], method='simplex')
            print(result.x)
            return result.x
        return None

    def Dispatch_active(self, node, mode, queue_p, queue_v):
        if (self.vehicle_attri[mode]['reb'] == 'active'):
            opt_queue_v = self.MaxWeight(node=node, queue=queue_p, server=queue_v)
            # normalize
            return opt_queue_v / (np.sum(opt_queue_v))
        else: 
            return None
        