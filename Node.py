
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
        self.arr_rate = 0.5
        self.dest = [d for d in graph_top]

        # default: exponential distribution
        # rate = ln(1-p) => p = 1-exp(-rate)
        random_dstr = np.random.uniform(low=0, high=1, size=(len(self.dest)))
        self.arr_rate_set = 1- np.exp( -self.arr_rate * random_dstr / np.sum(random_dstr) )
        # print(self.arr_rate_set, np.sum(self.arr_rate_set))

        for m in self.mode:
            self.vehicle[m] = {}
    
    def get_id(self):
        return self.id

    def get_location(self):
        return self.loc

    def get_mode(self):
        return self.mode

    def syn_time(self, time):
        self.time = time

    def check_accessiblity(self, mode):
        return (mode in self.mode)

    def new_passenger_generator(self, g):
        randomness = np.random.uniform(low=0, high=1, size=(len(self.dest)))
        node_cnt = 0
        for res in np.greater(self.arr_rate_set, randomness):
            if (res):
                dest = self.dest[node_cnt]
                pid = '{}{}{}'.format(self.id, dest, self.time)
                p = Passenger(pid=pid, ori=self.id, dest=dest, arr_time=self.time)
                print(pid, p.get_schdule(g))
            node_cnt = node_cnt+1

    def passenger_arrival(self, p):
        self.passenger.append(p)

    def passenger_leave(self, p):
        self.passenger.remove(p)