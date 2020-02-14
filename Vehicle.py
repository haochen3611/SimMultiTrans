#!/usr/bin/env python
# -*- coding: utf-8 -*-

class Vehicle(object):
    def __init__(self, vid, mode, ori):
        self.id = vid
        self.moed = mode

        self.cur_pos = ori
        self.seats = []
        
        self.waiting_sts = False        
        self.passenger_sts = False


    def update_location(self, loc):
        self.cur_pos = loc

    def get_passengers(self):
        return self.seats

    def pickup(self, p):
        self.seats.append(p)

    def dropoff(self):
        drop_list = []
        for p in self.seats:
            if (p.dropoff()):
                drop_list.append(p)
                self.seats.remove(p)
    

