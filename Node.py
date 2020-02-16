
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from Passenger import Passenger
from Vehicle import Vehicle

import numpy as np

class Node(object):
    def __init__(self, nid, graph_top):
        self.id = nid

        self.loc = (graph_top[nid]['locx'], graph_top[nid]['locy'])
        self.mode = graph_top[nid]['mode'].split(',')

        self.time = 0
        # print(self.id, self.loc, self.mode)
        self.passenger = []
        self.vehicle = {}

        # default arrival process
        self.arr_rate = np.random.uniform(0, 0.5, 1)
        self.dest = [d for d in graph_top]

        # random_dstr = np.random.uniform(low=0, high=1, size=(len(self.dest)))
        self.arr_prob_set = self.randon_exp_arrival_prob(len(graph_top))
        # print(self.arr_rate_set, np.sum(self.arr_rate_set))

        for m in self.mode:
            self.vehicle[m] = {}
    
    def get_id(self):
        return self.id

    def get_location(self):
        return self.loc

    def get_mode(self):
        return self.mode

    def get_passenger_queue(self):
        return self.passenger
    
    def get_vehicle_queue(self):
        return self.vehicle

    def syn_time(self, time):
        self.time = time

    def check_accessiblity(self, mode):
        return (mode in self.mode)

    def passenger_arrival(self, p):
        self.passenger.append(p)

    def passenger_leave(self, p):
        self.passenger.remove(p)

    def randon_exp_arrival_prob(self, size):
        # default: exponential distribution
        # rate = ln(1-p) => p = 1-exp(-rate)
        rd = np.random.uniform(0, 5, size)
        return 1- np.exp( - self.arr_rate* rd/np.sum(rd) )

    def new_passenger_generator(self, g):
        randomness = np.random.uniform(low=0, high=1, size=(len(self.dest)))

        pp = np.greater(self.arr_prob_set, randomness)
        for index, res in enumerate( pp ):
            if (res):
                dest = self.dest[index]
                pid = '{}{}{}'.format(self.id, dest, self.time)
                p = Passenger(pid=pid, ori=self.id, dest=dest, arr_time=self.time)
                # self.passenger.append(p)
                # print(pid, p.get_schdule(g))
                self.passenger_arrival(p)