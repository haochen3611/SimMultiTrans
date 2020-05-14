#!/usr/bin/env python
# -*- coding: utf-8 -*-
from .Passenger import Passenger
# from bin.Network.Vehicle import Vehicle
from .Road import Road

import numpy as np

import logging
import copy


class Node(object):
    """
    Use Node class to simulate the behaviors of passengers and vehicles in a node\\
    methods:\\
        passenger_arrive(p): a passenger arrived\\
        passenger_leave(p): a passenger left\\
        vehicle_arrive(v): a vehicle arrived\\
        vehicle_leave(v): a passenger left\\
        match_demands(attri): match demands and supplies based on attri(json)\\
        dispatch(reb_flow): rebalance vehicles based on reb_flow(np.arry)
    """

    def __init__(self, nid, graph_top):
        self.id = nid
        self.graph_top = graph_top

        self.lat = graph_top[nid]['lat']
        self.lon = graph_top[nid]['lon']
        self.mode = graph_top[nid]['mode'].split(',')

        self.road = {}
        for dest in graph_top[nid]['nei']:  # TODO: try separate this with the constructor
            # distance can be set by L1 norm
            L1dist = Haversine((self.lat, self.lon), (graph_top[dest]['lat'], graph_top[dest]['lon'])).meters
            if graph_top[nid]['nei'][dest]['dist'] <= 0.2 * L1dist:
                # dist = L1dist
                graph_top[nid]['nei'][dest]['dist'] = L1dist

            time = 0 if ('time' not in graph_top[nid]['nei'][dest]) else graph_top[nid]['nei'][dest]['time']
            r = Road(ori=nid, dest=dest, dist=graph_top[nid]['nei'][dest]['dist'], time=time)
            self.road[dest] = r

        self.time = 0

        # wait for child growing up to a passenger (exp time)
        self.child_passenger = [[]]

        self.passenger = {}
        self.total_p = 0
        self.total_served_p = 0

        self.vehicle = {}
        self.p_wait = {}
        for mode in self.mode:
            self.vehicle[mode] = []
            self.passenger[mode] = []
            self.p_wait[mode] = []

            # a copy of walk
        self.walk = None

        # default arrival process
        self.arr_rate = np.random.uniform(0, 0.5, 1)
        self.dest = [d for d in graph_top]

        self.park = []

        # random_dstr = np.random.uniform(low=0, high=1, size=(len(self.dest)))
        self.arr_prob_set = self.random_exp_arrival_prob(1, len(graph_top))
        # print(self.arr_rate_set, np.sum(self.arr_rate_set))

    def get_average_wait_time(self, mode):
        return 0 if mode not in self.p_wait or len(self.p_wait[mode]) == 0 else sum(self.p_wait[mode]) / len(
            self.p_wait[mode])

    def check_accessibility(self, mode):
        return mode in self.mode

    def set_arrival_rate(self, rate):
        assert len(rate) == len(self.arr_prob_set)

        self.arr_rate = np.sum(rate)
        self.arr_prob_set = self.exp_arrival_prob(rate)

    def set_walk(self, v):
        self.walk = copy.deepcopy(v)

    def passenger_arrive(self, p):
        # self.passenger.append(p)
        mode = p.get_waitingmode(self.id)
        # print(mode)
        if mode is not None:
            self.passenger[mode].append(p)
            # p.set_stoptime(self.time)
            p.stoptime = self.time
            # walk always available
            if mode == 'walk':
                v_walk = copy.deepcopy(self.walk)
                self.vehicle_arrive(v_walk)

    def passenger_leave(self, p):
        # self.passenger.remove(p)
        mode = p.get_waitingmode(self.id)
        if mode:
            # waittime = self.time - p.get_stoptime()
            waittime = self.time - p.stoptime
            self.p_wait[mode].append(waittime)
            self.passenger[mode].remove(p)

    def passengers_clear(self):
        not_served_p = 0
        for mode in self.passenger:
            for p in self.passenger[mode]:
                # waittime = self.time - p.get_stoptime()
                waittime = self.time - p.stoptime
                self.p_wait[mode].append(waittime)
                not_served_p += 1
        return not_served_p

    def vehicle_park(self, v, leavetime):
        self.park.append((v, leavetime))

    def vehicle_arrive(self, v):
        if v.mode != 'walk':
            logging.info(f'Time {self.time}: Vel {v.id} arrive at node {self.id}')
        self.vehicle[v.mode].append(v)
        v.loc = self.id
        p_list = v.dropoff()

        isVwalk = False

        for p in p_list:
            if p.dest == self.id:
                # self.passenger_leave(p)
                logging.info(f'Time {self.time}: Pas {p.id} arrived at destination {self.id} and quit')
                self.total_served_p += 1
                del p
                if v.mode == 'walk':
                    # passenger's foot leaves with himself!
                    # del v
                    isVwalk = True
            else:
                # self.passenger.append(p)
                self.passenger_arrive(p)
                logging.info(f'Time {self.time}: Pas {p.id} arrived at {self.id}')

                # if v arrive at final stop, then move to the park
        if not isVwalk:
            if v.finalstop(self.id):
                self.vehicle_leave(v)
                self.vehicle_park(v, self.time + v.park_time)
                logging.info(f'Time {self.time}: Vel {v.id} parking at {self.id}')

    def vehicle_leave(self, v):
        self.vehicle[v.mode].remove(v)
        if v.mode == 'walk':
            return
        if v.seats:
            logging.info(f'Time {self.time}: Vel {v.id} leave {self.id}')
        else:
            logging.info(f'Time {self.time}: Vel(empty) {v.id} leave {self.id}')

    def exp_arrival_prob(self, rate):
        # default: exponential distribution
        # rate = ln(1-p) => p = 1-exp(-rate)
        # exp distirbution is supported form 0 to inf
        # print(rate)
        prob = 1 - np.exp(-rate)
        prob[np.where(rate < 0)] = 0
        # print(prob)
        return prob

    def random_exp_arrival_prob(self, range_, size):
        """
        default: exponential distribution\\
        rate = ln(1-p) => p = 1-exp(-rate)\\
        args:\\
            range: np.array\\
            size: int
        """
        rd = np.random.uniform(0, range_, size)
        return 1 - np.exp(- self.arr_rate * rd / np.sum(rd))

    def passenger_generator(self, time_horizon):
        self.child_passenger = [[]] * time_horizon
        for time in range(time_horizon):
            self.child_passenger[time] = []
            randomness = np.random.uniform(low=0, high=1, size=(len(self.dest)))
            toss = np.greater(self.arr_prob_set, randomness)
            for index, p_arrive in enumerate(toss):
                if p_arrive:
                    dest = self.dest[index]
                    pid = f'{self.id}_{dest}_{time}'
                    self.child_passenger[time].append(Passenger(pid=pid, ori=self.id, dest=dest, arr_time=time))
                    self.total_p += 1

    def new_passenger_arrive(self, routing):
        if self.child_passenger[self.time]:
            for p in self.child_passenger[self.time]:
                p.get_schdule(routing)

                self.passenger_arrive(p)
                logging.info(f'Time {self.time}: Pas {p.id} arrived, ori={self.id}, dest={p.dest}')

    def match_demands(self, attri):
        # check if the vehicle leave the park
        while len(self.park) > 0:
            (v, time) = self.park[0]
            if time <= self.time:
                v.reverse_route(self.id)
                self.park.remove((v, time))
                self.vehicle_arrive(v)

        # if len(self.passenger.keys()) != 0:

        for mode in self.passenger:
            # for p in self.passenger[mode]:
            # while len(self.passenger[mode]) != 0:
            while self.passenger[mode]:
                p = self.passenger[mode][0]
                # if len(self.vehicle[mode]) == 0:
                if not self.vehicle[mode]:
                    break
                else:
                    # print(self.id, 'vehicle exist')
                    v = self.vehicle[mode][0]

                    if p.geton(self.id, v) and v.pickup(p):
                        v.set_destination(p.get_nextstop(self.id))
                        self.passenger_leave(p)

                        logging.info(f'Time {self.time}: Pas {p.id} takes {v.id} at node {self.id}')
                        # if vehicle is full, then move to road
                        if v.get_emptyseats() == 0 or v.type == 'priv':
                            self.vehicle_leave(v)
                            self.road[v.nextstop].arrive(v)

                            # all public trans will leave
        for mode in self.vehicle:
            # if ( attri[mode]['type'] == 'publ' and len(self.vehicle[mode]) != 0):
            if attri[mode]['type'] == 'publ' and self.vehicle[mode]:
                for v in self.vehicle[mode]:
                    v.set_destination(None)
                    self.vehicle_leave(v)
                    self.road[v.nextstop].arrive(v)

    def dispatch(self, reb_flow):
        # print(reb_flow)
        if not reb_flow:
            return
        for mode in reb_flow:
            if mode in self.vehicle and np.sum(reb_flow[mode]) != 0 and mode != 'walk':
                # only if more supplies
                # if len(self.passenger[mode]) == 0 and len(self.vehicle[mode]) > 0 and reb_flow[mode]['reb']:
                if not self.passenger[mode] and self.vehicle[mode] and reb_flow[mode]['reb']:
                    leave_list = []
                    stay_list = []

                    # set destination for all vehicles that will be dispatched
                    for v in self.vehicle[mode]:
                        if np.sum(reb_flow[mode]['p']) == 0:
                            return

                        dest = np.random.choice(reb_flow['nodes'], 1, p=reb_flow[mode]['p'])[0]
                        '''
                        if ( mode in self.graph_top[dest]['mode'] and dest != self.id ):
                            v.set_destination(dest)
                            self.vehicle_leave(v)
                            self.road[v.nextstop].arrive(v)
                            # logging.info(f'Time {self.time}: Vel {v.id} is dispatched from {self.id} to {dest}')
                        '''
                        if mode in self.graph_top[dest]['mode'] and dest != self.id:
                            v.set_destination(dest)
                            leave_list.append(v)

                    # dispatch vehicles
                    for v in leave_list:
                        self.vehicle_leave(v)
                        self.road[v.nextstop].arrive(v)


class Haversine:
    """
    use the haversine class to calculat1e the distance between
    two lon/lat coordnate pairs.
    output distance available in kilometers, meters, miles, and feet.
    example usage: Haversine([lon1,lat11],[lon2,lat12]).feet
    """

    def __init__(self, coord1, coord2):
        lon1, lat11 = coord1
        lon2, lat12 = coord2

        R = 6371000  # radius of Earth in meters
        phi_1 = np.radians(lat11)
        phi_2 = np.radians(lat12)

        delta_phi = np.radians(lat12 - lat11)
        delta_lambda = np.radians(lon2 - lon1)

        a = np.sin(delta_phi / 2.0) ** 2 + \
            np.cos(phi_1) * np.cos(phi_2) * \
            np.sin(delta_lambda / 2.0) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        self.meters = R * c  # output distance in meters
        self.km = self.meters / 1000.0  # output distance in kilometers
        self.miles = self.meters * 0.000621371  # output distance in miles
        self.feet = self.miles * 5280  # output distance in feet
