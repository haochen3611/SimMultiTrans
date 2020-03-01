#!/usr/bin/env python
# -*- coding: utf-8 -*-
# from Node import Node

import numpy as np
from scipy.optimize import linprog

class Rebalancing(object):
    def __init__(self, graph, vehicle_attri):
        self.graph = graph
        self.vehicle_attri = vehicle_attri


    def MaxWeight(self, queue, server):
        if (len(queue) == len(server)):
            A = np.ones(shape=(len(queue), 1))
            return linprog(c=queue, A_ub=A, b_ub=np.sum(server), bounds=[0, np.inf], method='simplex')
        return None

    def Dispatch_active(self, mode, queue_p, queue_v):
        if (self.vehicle_attri[mode]['reb'] == 'active'):
            opt_queue_v = self.MaxWeight(queue=queue_p, server=queue_v)