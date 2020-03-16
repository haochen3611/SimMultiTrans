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
        self.reb = attri['reb']

        self.route_set = []
        # public mode has own schedule
        if (self.type == 'publ'):
            self.route_set = self.route.split(',')
            self.loc = self.route_set[0]
            self.nextstop = self.route_set[1]
            # curloc = self.route_set.pop(0)
            # self.route_set.append(curloc)
            

    def get_id(self):
        return self.id

    def get_mode(self):
        return self.mode

    def get_velocity(self, unit):
        unit_trans = {
            'm/s': 0.44704,
            'mph': 1,
            'km/h': 1.60934,
        }
        if (unit not in unit_trans):
            unit = 'm/s'
        return (self.vel * unit_trans[unit])
    
    def get_type(self):
        return self.type

    def get_rebtype(self):
        return self.reb

    def reverse_route(self, loc):
        if (loc == self.route_set[-1]):
            self.route_set.reverse()
            # print(self.route_set)
            self.nextstop = self.route_set[1]

    def get_route(self):
        return self.route_set
    
    def set_destination(self, dest):
        if (self.type == 'priv'):
            self.nextstop = dest
        elif (self.type == 'publ'):
            loc_index = self.route_set.index(self.loc)
            self.nextstop = self.route_set[ loc_index+1 ]
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
        if (self.type == 'publ'):
            return True if (loc == self.route_set[-1]) else False       
        return False

    def get_parktime(self):
        interval_str = self.interval.split(' ')
        unit_trans = {
            'day': 60*60*24,
            'hour': 60*60,
            'min': 60,
            'sec': 1
        }
        # print(int(interval_str[0]) * unit_trans[interval_str[1]])
        return int(interval_str[0]) * unit_trans[interval_str[1]]

    def get_destination(self):
        return self.nextstop

    def update_location(self, loc):
        self.loc = loc            

    def get_passengers(self):
        return self.seats

    def get_emptyseats(self):
        return (self.cap - len(self.seats))
    
    def match_route(self, ori, dest):
        if (self.type == 'priv'):
            return True
        elif (ori == self.loc):
            bufroute = self.route_set[ self.route_set.index(ori): ]
            return True if dest in bufroute else False
        return False

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
    

