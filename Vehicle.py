#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools as itt

class Vehicle(object):
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
        self.route = attri['route']
        self.interval = attri['interval']

        self.route_set = []
        # public mode has own schedule
        if (self.type == 'publ'):
            self.route_set = self.route.split(',')
            self.loc = self.route_set[0]

    def get_id(self):
        return self.id

    def get_mode(self):
        return self.mode

    def get_velocity(self):
        return self.vel
    
    def get_type(self):
        return self.type

    def set_location(self, loc):
        self.loc = loc
    
    def set_destination(self, dest):
        if (self.type == 'priv'):
            self.nextstop = dest
        else:
            # self.nextstop = next(self.route_set)
            loc_index = self.route_set.index(self.loc)
            self.nextstop = self.route_set[ (loc_index+1)%len(self.route_set) ]

    def get_destination(self):
        return self.nextstop

    def update_location(self, loc):
        self.loc = loc

    def get_passengers(self):
        return self.seats

    def get_emptyseats(self):
        return (self.cap - len(self.seats))

    def pickup(self, p):
        if (self.get_emptyseats() != 0):
            self.seats.append(p)
            return True
        else:
            return False

    def dropoff(self):
        drop_list = []
        for p in self.seats:
            if (p.getoff(self.loc)):
                drop_list.append(p)
                self.seats.remove(p)
        return drop_list
    

