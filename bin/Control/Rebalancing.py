#!/usr/bin/env python
# -*- coding: utf-8 -*-
# from Node import Node

import numpy as np
from scipy.optimize import linprog

class Rebalancing(object):
    def __init__(self, graph, vehicle_attri):
        self.graph = graph
        self.vehicle_attri = vehicle_attri
        self.lazy = 0
        self.range = 0

        self.k_near_nei = {}
        self.size = len(self.graph.graph_top.keys())


        # np.set_printoptions(threshold=10000)

        self.costsensi_para = None

    def MaxWeight(self, queue, server):
        # weight
        c = np.kron( np.ones(self.size), -np.array(queue) )
        # print('c',c)

        # eq constraints
        A_eq = np.kron( np.eye(self.size), np.ones(self.size) )
        # b_eq = np.ones(shape=(self.size,1))
        b_eq = server

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
        # print(b_ub)
        
        # print('b',b_ub.T)
        # result = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[0, 1], method='simplex')
        result = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[0, max(server)], 
            method='interior-point', options={'lstsq': True, 'presolve': True})
        # print(result.x.reshape((self.size, self.size)))
        return result.x.reshape((self.size, self.size))


    def Perposion(self, queue, server):
        result = np.zeros(shape=(self.size,self.size))
        for index, node in enumerate(self.graph.graph_top):
            # b = np.zeros(self.size)
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

            asy = np.array(queue)
            sum_rate = np.sum(asy[k_near_list]) 
            if (sum_rate == 0):
                result[index,idnex] = 1
            else:
                rate = np.zeros(self.size)
                for k_near in k_near_list:
                    rate[k_near] = asy[k_near]/sum_rate
                result[index,:] = rate
        return result


    def MaxWeight_Heuristic(self, queue, server):
        result = np.zeros(shape=(self.size,self.size))
        for index, node in enumerate(self.graph.graph_top):
            # b = np.zeros(self.size)
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

            A_ub = np.eye(self.size)
            b_ub = np.zeros(shape=(self.size, 1))
            b_ub[k_near_list] = 1.0/np.sqrt(k)
            A_eq = np.ones(shape=(1, self.size))
            b_eq = 1
            c = -np.array(queue)
            sol = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[0, 1], method='interior-point')
            result[index,:] = sol.x            
        return result


    def CostSensitive(self, mode, queue, server):
        result = np.zeros(shape=(self.size,self.size))

        # set linear parameter (c, A_ub)
        if (self.costsensi_para == None):
            self.costsensi_para = {}
            T = np.zeros(shape=(self.size,self.size))
            for oi, ori in enumerate(self.graph.graph_top):
                '''
                T[index, :] = np.array( 
                    [ self.graph.graph_top[node]['nei'][dest]['dist']
                        for dest in self.graph.graph_top if (dest in self.graph.graph_top[node]['nei']) else 0 ] 
                    )
                '''
                for di, dest in enumerate(self.graph.graph_top):
                    T[oi, di] = self.graph.graph_top[ori]['nei'][dest]['dist'] if (dest in self.graph.graph_top[ori]['nei']) else 0
            c = T.reshape(self.size**2, 1)

            A_ub = np.zeros(shape=(self.size, self.size**2))
            for row in range(self.size):
                r = np.zeros(self.size)
                r[row] = -1
                A_ub[row,:] = np.kron( np.ones(self.size), r )
                nr = np.ones(self.size)
                nr[row] = 0
                A_ub[row,row*self.size:(row+1)*self.size] = nr
            self.costsensi_para['c'] = c
            self.costsensi_para['Aub'] = A_ub
        else:
            c = self.costsensi_para['c']            
            A_ub = self.costsensi_para['Aub']

        b_ub = np.array(server) - float(self.vehicle_attri[mode]['total'] - sum(queue))/self.size
        # print(c,A_ub)
        result = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=[0, max(server)], 
            method='interior-point')
        # result = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=[0, max(server)], method='simplex')
        return result.x.reshape((self.size, self.size))

    def Simplified_CostSensitive(self, mode, queue, server):
        result = np.zeros(shape=(self.size,self.size))

        # eq constraints
        A_eq = np.kron( np.eye(self.size), np.ones(self.size) )
        # b_eq = np.ones(shape=(self.size,1))
        b_eq = server

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
                '''
                T[index, :] = np.array( 
                    [ self.graph.graph_top[node]['nei'][dest]['dist']
                        for dest in self.graph.graph_top if (dest in self.graph.graph_top[node]['nei']) else 0 ] 
                    )
                '''
                for di, dest in enumerate(self.graph.graph_top):
                    T[oi, di] = self.graph.graph_top[ori]['nei'][dest]['dist'] if (dest in self.graph.graph_top[ori]['nei']) else 0
            c = T.reshape(self.size**2, 1)
            self.costsensi_para['c'] = c
        else:
            c = self.costsensi_para['c']            
            # A_ub = self.costsensi_para['Aub']

        # make the constraints be part of objective function
        # print(c)
        c = c.T*10/np.max(c) + np.kron( np.ones(self.size), np.array(server)-np.array(queue) )
        

        # b_ub = np.array(server) - float(self.vehicle_attri[mode]['total'] - sum(queue))/self.size
        # print(c,A_ub)
        result = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[0, max(server)], 
            method='interior-point', options={'lstsq': True, 'presolve': True})
        # result = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=[0, max(server)], method='simplex')
        return result.x.reshape((self.size, self.size))

    
    def Dispatch_active(self, mode, queue_p, queue_v):
        if (self.vehicle_attri[mode]['reb'] == 'active' 
            and np.sum(queue_p) != 0 and len(queue_p) == len(queue_v) ):

            # opt_trans = self.MaxWeight(queue=queue_p, server=queue_v)
            # opt_trans = self.Perposion(queue=queue_p, server=queue_v)
            # opt_trans = self.MaxWeight_Heuristic(queue=queue_p, server=queue_v)
            # opt_trans = self.CostSensitive(mode=mode ,queue=queue_p, server=queue_v)
            opt_trans = self.Simplified_CostSensitive(mode=mode ,queue=queue_p, server=queue_v)
            
            opt_flow = {}
            for index, node in enumerate(self.graph.graph_top):
                # print(index*len(queue_p), (index+1)*len(queue_p)-1)

                # add lazy
                opt_trans[index,index] += self.lazy

                opt_flow[node] = opt_trans[index,:]
                flow_sum = np.sum(opt_flow[node])
                
                if (flow_sum != 1 and flow_sum != 0):
                    # print(flow_sum)
                    opt_flow[node] = np.array( [flow/flow_sum for flow in opt_flow[node]] )

            return opt_flow, True  
        else: 
            return None, False
        