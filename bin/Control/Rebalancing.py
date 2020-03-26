#!/usr/bin/env python
# -*- coding: utf-8 -*-
# from Node import Node

import numpy as np
from scipy.optimize import linprog

class Rebalancing(object):
    def __init__(self, graph, vehicle_attri):
        self.graph = graph
        self.vehicle_attri = vehicle_attri
        self.lazy = 10
        self.range = 20

        self.k_near_nei = {}
        np.set_printoptions(threshold=10000)

    def MaxWeight(self, queue, server):
        # print(len(queue))
        if (np.sum(queue) != 0 and len(queue) == len(server)):
            # weight
            c = np.kron( np.ones(len(queue)), -np.array(queue) )
            # print('c',c)

            # eq constraints
            A_eq = np.kron( np.eye(len(queue)), np.ones(len(queue)) )
            b_eq = np.ones(shape=(len(queue),1))

            # ineq constraints
            b_ub = np.zeros(shape=(len(queue), len(queue)))
            for index, node in enumerate(self.graph.graph_top):
                b = np.zeros(len(queue))
                if (node not in self.k_near_nei):
                    dist_list = np.array( [ self.graph.graph_top[node]['nei'][dest]['dist']
                    for dest in self.graph.graph_top[node]['nei'] ] )
                    k = self.range
                    if (k > len(queue)-1):
                        k = len(queue)-1
                    # k = len(queue)-1
                    k_near_list = dist_list.argsort()[:k]
                else:
                    k_near_list = self.k_near_nei[node]
                b[k_near_list] = 1.0
                b_ub[[index],:] = b.T
            
            b_ub = b_ub.reshape((len(queue)**2, 1))
            A_ub = np.eye( len(b_ub) )
            # print(A_ub)
            
            # print('b',b_ub.T)
            # result = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[0, 1], method='simplex')
            result = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[0, 1], method='interior-point')
            # print(result)
            return result.x
        return np.zeros(len(queue)**2)

    def Perposion(self, node, queue, server):
        if (np.sum(queue) != 0 and len(queue) == len(server)):
            dist_list = np.array( [ self.graph.graph_top[node]['nei'][dest]['dist']
                for dest in self.graph.graph_top[node]['nei'] ] )
            
            k = self.range
            if (k > len(queue)-1):
                k = len(queue)-1
            k_near_list = dist_list.argsort()[:k]
            
            asy = np.array(queue)
            sum_rate = np.sum(asy[k_near_list]) 
            if (sum_rate == 0):
                return np.zeros(len(queue))

            rate = np.zeros(len(queue))
            for k_near in k_near_list:
                rate[k_near] = asy[k_near]/sum_rate
            return rate
        return np.zeros(len(queue))
    

    def Dispatch_active(self, mode, queue_p, queue_v):
        if (self.vehicle_attri[mode]['reb'] == 'active'):
            opt_trans = self.MaxWeight(queue=queue_p, server=queue_v)
            # print(len(opt_trans)/len(queue_p))
            opt_trans = opt_trans.reshape((len(queue_p), len(queue_p)))
            opt_flow = {}
            for index, node in enumerate(self.graph.graph_top):
                # print(index*len(queue_p), (index+1)*len(queue_p)-1)
                opt_flow[node] = opt_trans[index,:]
                flow_sum = np.sum(opt_flow[node])
                
                if (flow_sum != 1 and flow_sum != 0):
                    # print(flow_sum)
                    opt_flow[node] = np.array( [flow/flow_sum for flow in opt_flow[node]] )
                # print(opt_flow[node])
                # opt_flow[node] = np.delete(opt_flow[node], [index])
            return opt_flow, True  
        else: 
            return None, False
        