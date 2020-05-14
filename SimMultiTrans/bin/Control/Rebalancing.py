#!/usr/bin/env python
# -*- coding: utf-8 -*-
# from Node import Node

import numpy as np
from scipy.optimize import linprog

import warnings
import sys
import logging


logger = logging.getLogger(__name__)


class Rebalancing(object):
    def __init__(self, graph, vehicle_attri):
        self.graph = graph
        self.vehicle_attri = vehicle_attri
        self.lazy = 0
        self.range = 0

        self.k_near_nei = {}
        self.size = len(self.graph.graph_top.keys())

        self.policies = {
            'MaxWeight': self.MaxWeight,
            'Simplified_MaxWeight': self.Simplified_MaxWeight,
            'Proportional': self.Proportional,
            'CostSensitive': self.CostSensitive,
            'Simplified_CostSensitive': self.Simplified_CostSensitive,
            'None': self.no_rebalance
        }
        # default policy
        self.policy = 'Simplified_MaxWeight'

        # np.set_printoptions(threshold=10000)

        self.costsensi_para = None
        warnings.filterwarnings('ignore')
        # np.set_printoptions(threshold=sys.maxsize)

    def MaxWeight(self, mode, queue, server):

        result = np.zeros(shape=(self.size, self.size))
        for index, node in enumerate(self.graph.graph_top):
            k = self.range
            if k > self.size - 1:
                k = self.size - 1
            if node not in self.k_near_nei:
                dist_list = np.array([self.graph.graph_top[node]['nei'][dest]['dist']
                                      for dest in self.graph.graph_top[node]['nei']])
                self.k_near_nei[node] = dist_list.argsort()[:k]
        asy = np.array(server)
        for di, dest in enumerate(self.graph.graph_top):
            while queue[di] > 0:
                queue[di] -= 1
                max_nei = np.argmax(asy[self.k_near_nei[dest]])
                ori = [oi for oi, ori in enumerate(asy)
                       if asy[oi] == asy[self.k_near_nei[dest]][max_nei] and oi in self.k_near_nei[dest]]
                result[ori[0]][di] += 1
                asy[ori[0]] -= 1
        return result

    def Proportional(self, mode, queue, server):
        result = np.zeros(shape=(self.size, self.size))
        for index, node in enumerate(self.graph.graph_top):
            # b = np.zeros(self.size)
            k = self.range
            if k > self.size - 1:
                k = self.size - 1
            if node not in self.k_near_nei:
                dist_list = np.array([self.graph.graph_top[node]['nei'][dest]['dist']
                                      for dest in self.graph.graph_top[node]['nei']])

                # k = self.size-1
                k_near_list = dist_list.argsort()[:k]
                self.k_near_nei[node] = k_near_list
            else:
                k_near_list = self.k_near_nei[node]

            asy = np.array(queue)
            sum_rate = np.sum(asy[k_near_list])
            if sum_rate == 0:
                result[index, index] = 1
            else:
                rate = np.zeros(self.size)
                for k_near in k_near_list:
                    rate[k_near] = asy[k_near] / sum_rate
                result[index, :] = rate
        return result

    def Simplified_MaxWeight(self, mode, queue, server):
        result = np.zeros(shape=(self.size, self.size))
        for index, node in enumerate(self.graph.graph_top):
            # b = np.zeros(self.size)
            k = self.range
            if k > self.size - 1:
                k = self.size - 1
            if node not in self.k_near_nei:
                dist_list = np.array([self.graph.graph_top[node]['nei'][dest]['dist']
                                      for dest in self.graph.graph_top[node]['nei']])

                # k = self.size-1
                k_near_list = dist_list.argsort()[:k]
                self.k_near_nei[node] = k_near_list
            else:
                k_near_list = self.k_near_nei[node]

            A_ub = np.eye(self.size)
            b_ub = np.zeros(shape=(self.size, 1))
            b_ub[k_near_list] = 1.0 / np.sqrt(k)
            A_eq = np.ones(shape=(1, self.size))
            b_eq = 1
            c = -np.array(queue)
            sol = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[0, 1], method='interior-point')
            result[index, :] = sol.x
        return result

    def CostSensitive(self, mode, queue, server):
        result = np.zeros(shape=(self.size, self.size))

        # set linear parameter (c, A_ub)
        if self.costsensi_para is None:
            self.costsensi_para = {}
            T = np.zeros(shape=(self.size, self.size))
            for oi, ori in enumerate(self.graph.graph_top):
                '''
                T[index, :] = np.array( 
                    [ self.graph.graph_top[node]['nei'][dest]['dist']
                        for dest in self.graph.graph_top if (dest in self.graph.graph_top[node]['nei']) else 0 ] 
                    )
                '''
                for di, dest in enumerate(self.graph.graph_top):
                    T[oi, di] = self.graph.graph_top[ori]['nei'][dest]['dist'] if (
                                dest in self.graph.graph_top[ori]['nei']) else 0
            c = T.reshape(self.size ** 2, 1)

            A_ub = np.zeros(shape=(self.size, self.size ** 2))
            for row in range(self.size):
                r = np.zeros(self.size)
                r[row] = -1
                A_ub[row, :] = np.kron(np.ones(self.size), r)
                nr = np.ones(self.size)
                nr[row] = 0
                A_ub[row, row * self.size:(row + 1) * self.size] = nr
            self.costsensi_para['c'] = c
            self.costsensi_para['Aub'] = A_ub
        else:
            c = self.costsensi_para['c']
            A_ub = self.costsensi_para['Aub']

        b_ub = np.array(server) - float(self.vehicle_attri[mode]['total'] - sum(queue)) / self.size
        # print(c,A_ub)
        result = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=[0, max(server)],
                         method='interior-point')
        # result = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=[0, max(server)], method='simplex')
        return result.x.reshape((self.size, self.size))

    def Simplified_CostSensitive(self, mode, queue, server):
        result = np.zeros(shape=(self.size, self.size))
        '''
        # eq constraints
        A_eq = np.kron( np.eye(self.size), np.ones(self.size) )
        b_eq = np.ones(shape=(self.size,1))
        # b_eq = server

        # ineq constraints
        b_ub = np.zeros(shape=(self.size, self.size))
        for index, node in enumerate(self.graph.graph_top):
            b = np.zeros(self.size)
            if (node not in self.k_near_nei):
                dist_list = np.array( [ self.graph.graph_top[node]['nei'][dest]['dist']
                for dest in self.graph.graph_top[node]['nei'] ] )
                k = self.range
                if (k > self.size-1):
                    k = self.size-1
                # k = self.size-1
                k_near_list = dist_list.argsort()[:k]
            else:
                k_near_list = self.k_near_nei[node]
            b[k_near_list] = server[index]
            b_ub[[index],:] = b.T
            # print(b.T)

        b_ub = b_ub.reshape((self.size**2, 1))
        A_ub = np.eye( len(b_ub) )

        # set linear parameter (c, A_ub)
        if (self.costsensi_para == None):
            self.costsensi_para = {}
            T = np.zeros(shape=(self.size,self.size))
            for oi, ori in enumerate(self.graph.graph_top):

                # T[index, :] = np.array( 
                #     [ self.graph.graph_top[node]['nei'][dest]['dist']
                #         for dest in self.graph.graph_top if (dest in self.graph.graph_top[node]['nei']) else 0 ] 
                #     )

                for di, dest in enumerate(self.graph.graph_top):
                    T[oi, di] = self.graph.graph_top[ori]['nei'][dest]['dist'] if (dest in self.graph.graph_top[ori]['nei']) else 0
            c = T.reshape(self.size**2, 1)
            self.costsensi_para['c'] = c
        else:
            c = self.costsensi_para['c']            
            # A_ub = self.costsensi_para['Aub']

        # make the constraints be part of objective function
        # print(c)
        c = c.T/np.max(c) + np.kron( np.ones(self.size), np.array(server)-np.array(queue) )
        # print(c)

        # b_ub = np.array(server) - float(self.vehicle_attri[mode]['total'] - sum(queue))/self.size
        # print(c,A_ub)
        result = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
            method='simplex', options={'lstsq': True, 'presolve': True})
        # result = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=[0, max(server)], method='simplex')
        # print(result.x)
        '''
        # set linear parameter (c, A_ub)
        if self.costsensi_para is None:
            self.costsensi_para = {}
            T = np.zeros(shape=(self.size, self.size))
            for oi, ori in enumerate(self.graph.graph_top):
                '''
                T[index, :] = np.array( 
                    [ self.graph.graph_top[node]['nei'][dest]['dist']
                        for dest in self.graph.graph_top if (dest in self.graph.graph_top[node]['nei']) else 0 ] 
                    )
                '''
                for di, dest in enumerate(self.graph.graph_top):
                    T[oi, di] = self.graph.graph_top[ori]['nei'][dest]['dist'] if (
                                dest in self.graph.graph_top[ori]['nei']) else 0
            c = T.reshape(self.size ** 2, 1)

            # A_ub = np.zeros(shape=(self.size, self.size**2))
            for row in range(self.size):
                r = np.zeros(self.size)
                r[row] = -1
                # A_ub[row,:] = np.kron( np.ones(self.size), r )
                nr = np.ones(self.size)
                nr[row] = 0
                # A_ub[row,row*self.size:(row+1)*self.size] = nr
            self.costsensi_para['c'] = c
            # self.costsensi_para['Aub'] = A_ub
        else:
            c = self.costsensi_para['c']
            # A_ub = self.costsensi_para['Aub']
        c = c.T / np.max(c) + np.kron(np.ones(self.size), np.array(server) - np.array(queue))
        # b_ub = np.array(server) - float(self.vehicle_attri[mode]['total'] - sum(queue))/self.size
        # print(c,A_ub)
        result = linprog(c=c, bounds=[0, max(server)],
                         method='interior-point')
        return result.x.reshape((self.size, self.size))

    def no_rebalance(self, mode, queue, server):
        return np.zeros(shape=(self.size, self.size))

    def set_parameters(self, lazy=0, v_range=0):
        """
        Set the parameters for the rebalancing policy\\
        arge:\\
            lazy: weight the probability that a vehicle stays at current node\\
            v_range: the number of nearest neighbourhoods that a vehicle can be dispathced
        """
        self.lazy = lazy
        self.range = v_range
        logger.info(f'Rebalancing Policy Parameters: lazy={lazy}, range={v_range}')

    def set_policy(self, policy):
        if policy in self.policies:
            self.policy = policy
            logger.info(f'Rebalancing Policy: {policy}')
        else:
            # cur_policy_name = [name for name in self.policies if self.policies[name] == self.policy]
            logger.info(f'{policy} is unavailable. Will use {self.policy}.')

    def Dispatch_active(self, mode, queue_p, queue_v):
        if self.vehicle_attri[mode]['reb'] == 'active' and np.sum(queue_p) != 0 and len(queue_p) == len(queue_v):

            opt_trans = self.policies[self.policy](mode=mode, queue=queue_p, server=queue_v)

            opt_flow = {}
            for index, node in enumerate(self.graph.graph_top):
                # add lazy
                opt_trans[index, index] += self.lazy

                opt_flow[node] = opt_trans[index, :]
                flow_sum = np.sum(opt_flow[node])

                if flow_sum != 1 and flow_sum != 0:
                    # print(flow_sum)
                    opt_flow[node] = np.array([flow / float(flow_sum) for flow in opt_flow[node]])

                if flow_sum == 0:
                    return opt_flow, False
            # print(opt_flow)
            return opt_flow, True
        else:
            return None, False
