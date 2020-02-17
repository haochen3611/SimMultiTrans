
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from Passenger import Passenger
from Vehicle import Vehicle
from Road import Road

import numpy as np

import logging

class Node(object):
    def __init__(self, nid, graph_top):
        self.id = nid

        self.loc = (graph_top[nid]['locx'], graph_top[nid]['locy'])
        self.mode = graph_top[nid]['mode'].split(',')

        self.road = {}
        for dest in graph_top[nid]['nei']:
            r = Road(ori=nid, dest=dest, dist=graph_top[nid]['nei'][dest]['dist'])
            self.road[dest] = r

        self.time = 0
        # print(self.id, self.loc, self.mode)
        self.passenger = []
        self.vehicle = {}

        # default arrival process
        self.arr_rate = np.random.uniform(0, 0.5, 1)
        self.dest = [d for d in graph_top]

        # random_dstr = np.random.uniform(low=0, high=1, size=(len(self.dest)))
        self.arr_prob_set = self.random_exp_arrival_prob(5, len(graph_top))
        # print(self.arr_rate_set, np.sum(self.arr_rate_set))

        for mode in self.mode:
            self.vehicle[mode] = []
    
    def get_id(self):
        return self.id

    def get_location(self):
        return self.loc

    def get_mode(self):
        return self.mode

    def get_road(self):
        return self.road

    def get_passenger_queue(self):
        return self.passenger
    
    def get_vehicle_queue(self, mode):
        return self.vehicle[mode]

    def syn_time(self, time):
        self.time = time

    def check_accessiblity(self, mode):
        return (mode in self.mode)

    def set_arrival_rate(self, rate):
        if (len(rate) != len(self.arr_prob_set)):
            print('Error arrival rate')
            return
        self.arr_rate = np.sum(rate)
        self.arr_prob_set = self.exp_arrival_prob(rate)

    def passenger_arrival(self, p):
        self.passenger.append(p)

    def passenger_leave(self, p):
        self.passenger.remove(p)

    def vehilce_arrival(self, v):
        logging.info('Time {}: {} arrive at node {}'.format(
            self.time, v.get_id(), self.id))
        self.vehicle[v.get_mode()].append(v)
        v.set_location(self.id)
        p_list = v.dropoff()            

        for p in p_list:
            if (p.get_odpair()[1] == self.id):
                # self.passenger_leave(p)
                logging.info('Time {}: Pas {} arrived at destination {} and quit'.format(
                    self.time, p.get_id(), self.id))
            else:
                self.passenger.append(p)
                logging.info('Time {}: Pas {} arrived at {}'.format(
                    self.time, v.get_id(), self.id))

    def vehilce_leave(self, v):
        self.vehicle[v.get_mode()].remove(v)
        logging.info('Time {}: Vel {} leave {}'.format(
                    self.time, v.get_id(), self.id))

    def exp_arrival_prob(self, rate):
        # default: exponential distribution
        # rate = ln(1-p) => p = 1-exp(-rate)
        return 1- np.exp( - rate )

    def random_exp_arrival_prob(self, range, size):
        # default: exponential distribution
        # rate = ln(1-p) => p = 1-exp(-rate)
        rd = np.random.uniform(0, range, size)
        return 1- np.exp( - self.arr_rate* rd/np.sum(rd) )

    def new_passenger_arrival(self, g):
        randomness = np.random.uniform(low=0, high=1, size=(len(self.dest)))

        pp = np.greater(self.arr_prob_set, randomness)
        for index, res in enumerate( pp ):
            if (res):
                dest = self.dest[index]
                pid = '{}{}_{}'.format(self.id, dest, self.time)
                p = Passenger(pid=pid, ori=self.id, dest=dest, arr_time=self.time)
                p.get_schdule(g)
                # self.passenger.append(p)
                # print(pid, p.get_schdule(g))
                self.passenger_arrival(p)
                logging.info('Time {}: Pas {} arrived, ori={}, dest={}'.format(
                    self.time, pid, self.id, dest))

    def match_demands(self, attri):
        if (len(self.passenger) != 0):
            for p in self.passenger:
                mode = p.get_waitingmode(self.id)
                if ( len(self.vehicle[mode]) != 0 ):
                    v = self.vehicle[mode][0]
                    if (v.pickup(p)):
                        v.set_destination(p.get_nextstop(self.id))
                        self.passenger_leave(p)
                        logging.info('Time {}: Pas {} takes {}'.format(
                            self.time, p.get_id(), v.get_id()))

                        # if vehicle is full, then move to road
                        if (v.get_emptyseats() == 0):
                            self.vehilce_leave(v)
                            self.road[v.get_destination()].arrive(v)
        
        # all public trans will leave
        for mode in self.vehicle:
            if ( attri[mode]['type'] == 'publ' and len(self.vehicle[mode]) != 0):
                for v in self.vehicle[mode]:
                    v.set_destination(None)
                    self.vehilce_leave(v)
                    self.road[v.get_destination()].arrive(v)
        
                

