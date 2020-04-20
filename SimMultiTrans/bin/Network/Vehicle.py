#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools as itt


class Vehicle(object):
    """
    Use Vehicle class to simulate the behaviors of vehicles\\
    methods:\\
        set_attri(attri): set the attribution to a vehicle\\
        set_destination(dest): set the destination\\
        match_route(ori, dest): if the o-d pair match the vehicle\\
        vehicle_leave(v): a passenger left\\
        pickup(p): pick up a passenger\\
        dropoff(): drop off all the passengers who want to get off
    """

    def __init__(self, vid, mode, loc):
        self.id = vid
        self.mode = mode

        self.loc = loc
        self.seats = []

        self.nextstop = loc

    def set_attri(self, attri):
        self.vel = attri['vel']
        self.cap = attri['cap']
        self.pri_mtd = attri['pri_mtd']
        self.pri = attri['pri']
        self.type = attri['type']
        self.onroad = attri['onroad']
        self.route = attri['route']
        self.interval = attri['interval']
        interval_str = self.interval.split(' ')
        unit_trans = {
            'day': 60 * 60 * 24,
            'hour': 60 * 60,
            'min': 60,
            'sec': 1
        }
        # print(int(interval_str[0]) * unit_trans[interval_str[1]])
        self.park_time = int(interval_str[0]) * unit_trans[interval_str[1]]

        self.reb = attri['reb']

        self.route_set = []
        # public mode has own schedule
        if self.type == 'publ':
            self.route_set = self.route.split(',')
            self.loc = self.route_set[0]
            self.nextstop = self.route_set[1]
            # curloc = self.route_set.pop(0)
            # self.route_set.append(curloc)

    def get_velocity(self, unit):
        unit_trans = {
            'm/s': 0.44704,
            'mph': 1,
            'km/h': 1.60934,
        }
        if unit not in unit_trans:
            unit = 'm/s'
        return self.vel * unit_trans[unit]

    def reverse_route(self, loc):
        """
        Reverse the route for public modes
        """
        if loc == self.route_set[-1]:
            self.route_set.reverse()
            # print(self.route_set)
            self.nextstop = self.route_set[1]

    def set_destination(self, dest):
        """
        Set destination to vehicle:\\
        Private vehicle's destination is oriented by the passenger\\
        Public vehicle's destination is oriented by its route
        """
        if self.type == 'priv':
            self.nextstop = dest
        elif self.type == 'publ':
            loc_index = self.route_set.index(self.loc)
            self.nextstop = self.route_set[loc_index + 1]
            '''
            if (self.loc != self.route_set[-1]):
                loc_index = self.route_set.index(self.loc)
                # self.nextstop = self.route_set[ (loc_index+1)%len(self.route_set) ]
                # self.nextstop = self.route_set[0]
                # curloc = self.route_set.pop(0)
                # self.route_set.append(curloc)
                self.nextstop = self.route_set[ loc_index+1 ]
            else:
                self.nextstop = None
            '''

    def finalstop(self, loc):
        """
        If it is the final stop of a public vehicle
        """
        if self.type == 'publ':
            return True if (loc == self.route_set[-1]) else False
        return False

    def get_emptyseats(self):
        return self.cap - len(self.seats)

    def get_occupiedseats(self):
        return len(self.seats)

    def match_route(self, ori, dest):
        """
        If the route of the vehicle is approperiate to the o-d pair
        """
        if self.type == 'priv':
            return True
        elif ori == self.loc:
            bufroute = self.route_set[self.route_set.index(ori):]
            return True if dest in bufroute else False
        return False

    def pickup(self, p):
        """
        Pick up a passenger (return False if the passenger takes a wrong vehicle)
        """
        if self.get_emptyseats() != 0:
            self.seats.append(p)
            # print(self.seats)
            return True
        else:
            return False

    def dropoff(self):
        """
        Drop off passengers (return a list of passengers)\\
        If this is walk, then drop off the only one passenger\\
        If the vehicle type is private, then drop off all passengers\\
        If the vehicle type is public, then drop off the passengers who want to get off
        """
        if self.type == 'priv' or self.mode == 'walk':
            drop_list = self.seats
            self.seats = []
        else:
            drop_list = []
            stay_list = []
            for p in self.seats:
                if p.dest == self.loc:
                    drop_list.append(p)
                elif p.getoff(self.loc):
                    drop_list.append(p)
                else:
                    stay_list.append(p)
            self.seats = stay_list
        return drop_list

