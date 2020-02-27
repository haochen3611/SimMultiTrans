
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from Passenger import Passenger
from Vehicle import Vehicle
from Road import Road

from Converter import Haversine

import numpy as np


import logging

class Node(object):
    def __init__(self, nid, graph_top):
        self.id = nid

        self.loc = (graph_top[nid]['lat'], graph_top[nid]['long'])
        self.mode = graph_top[nid]['mode'].split(',')

        self.road = {}
        for dest in graph_top[nid]['nei']:
            # distance can be set by L1 norm
            dist = graph_top[nid]['nei'][dest]['dist']
            # L1dist = np.abs(self.loc[0] - graph_top[dest]['lat']) + np.abs(self.loc[1] - graph_top[dest]['long'])
            L1dist = Haversine( (self.loc[0], self.loc[1]), (graph_top[dest]['lat'], graph_top[dest]['long']) ).meters
            if (dist <= 0.2* L1dist):
                dist = L1dist
            r = Road(ori=nid, dest=dest, dist=dist)
            self.road[dest] = r

        self.time = 0
        # print(self.id, self.loc, self.mode)
        # self.passenger = []
        self.passenger = {}
        self.vehicle = {}
        for mode in self.mode:
            self.vehicle[mode] = []
            self.passenger[mode] = []

        # default arrival process
        self.arr_rate = np.random.uniform(0, 0.5, 1)
        self.dest = [d for d in graph_top]

        self.park = []

        # random_dstr = np.random.uniform(low=0, high=1, size=(len(self.dest)))
        self.arr_prob_set = self.random_exp_arrival_prob(5, len(graph_top))
        # print(self.arr_rate_set, np.sum(self.arr_rate_set))

        
    
    def get_id(self):
        return self.id

    def get_location(self):
        return self.loc

    def get_mode(self):
        return self.mode

    def get_road(self):
        return self.road

    def get_passenger_queue(self, mode):
        # return self.passenger
        return self.passenger[mode]
    
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

    def passenger_arrive(self, p):
        # self.passenger.append(p)
        mode = p.get_waitingmode(self.id)
        # print(mode)
        if (mode != None):
            self.passenger[mode].append(p)

    def passenger_leave(self, p):
        # self.passenger.remove(p)
        mode = p.get_waitingmode(self.id)
        if (mode != None):
            self.passenger[mode].remove(p)

    def vehicle_park(self, v, leavetime):
        self.park.append( (v, leavetime) )

    def vehicle_arrive(self, v):
        logging.info('Time {}: Vel {} arrive at node {}'.format(
            self.time, v.get_id(), self.id))
        self.vehicle[v.get_mode()].append(v)
        v.update_location(self.id)
        p_list = v.dropoff()            

        for p in p_list:
            if (p.get_odpair()[1] == self.id):
                # self.passenger_leave(p)
                logging.info('Time {}: Pas {} arrived at destination {} and quit'.format(
                    self.time, p.get_id(), self.id))
            else:
                # self.passenger.append(p)
                self.passenger_arrive(p)
                logging.info('Time {}: Pas {} arrived at {}'.format(
                    self.time, p.get_id(), self.id))
        
        # if v arrive at final stop, then move to the park
        if (v.finalstop(self.id)):
            self.vehilce_leave(v)
            self.vehicle_park(v, self.time+ v.get_parktime())
            logging.info('Time {}: Vel {} parking at {}'.format(
                self.time, v.get_id(), self.id))

    def vehilce_leave(self, v):
        self.vehicle[v.get_mode()].remove(v)
        logging.info('Time {}: Vel {} leave {}'.format(
                    self.time, v.get_id(), self.id))

    def exp_arrival_prob(self, rate):
        # default: exponential distribution
        # rate = ln(1-p) => p = 1-exp(-rate)
        return 1- np.exp( -rate )

    def random_exp_arrival_prob(self, range, size):
        # default: exponential distribution
        # rate = ln(1-p) => p = 1-exp(-rate)
        rd = np.random.uniform(0, range, size)
        return 1- np.exp( - self.arr_rate* rd/np.sum(rd) )

    def new_passenger_arrive(self, routing):
        randomness = np.random.uniform(low=0, high=1, size=(len(self.dest)))

        pp_toss = np.greater(self.arr_prob_set, randomness)
        for index, res in enumerate(pp_toss):
            if (res):
                dest = self.dest[index]
                pid = '{}{}_{}'.format(self.id, dest, self.time)
                p = Passenger(pid=pid, ori=self.id, dest=dest, arr_time=self.time)

                # random pick a routing policy
                routing_method = np.random.choice(routing.get_methods(), 1)[0]
                p.get_schdule(routing, routing_method)
                # print('new passenger arrived')
                # print(p.get_id(), p.get_odpair())
                # print('get sch: ', p.get_schdule(routing))
                # self.passenger.append(p)
                # print(pid, p.get_schdule(g))
                self.passenger_arrive(p)
                logging.info('Time {}: Pas {} arrived, ori={}, dest={}'.format(
                    self.time, pid, self.id, dest))

    def match_demands(self, attri):
        # check if the vehicle leave the park
        for (v, time) in self.park:
            if (time <= self.time):
                v.reverse_route(self.id)
                self.park.remove( (v, time) )
                self.vehicle_arrive(v)

        if ( len(self.passenger) != 0 ):
            for mode in self.passenger:
                for p in self.passenger[mode]:
                    if ( len(self.vehicle[mode]) != 0 ):
                        v = self.vehicle[mode][0]

                        if (v.pickup(p) and p.geton(self.id, v)):
                            v.set_destination(p.get_nextstop(self.id))
                            self.passenger_leave(p)
                            logging.info('Time {}: Pas {} takes {} at node {}'.format(
                                self.time, p.get_id(), v.get_id(), self.id))

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
    
                

