
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from Passenger import Passenger
from Vehicle import Vehicle

import numpy as np

import logging

class Road(object):
    def __init__(self, ori, dest, dist):
        self.ori = ori
        self.dest = dest
        self.dist = dist
        self.time = 0
        
        self.vehicle = []
    
    def get_id(self):
        return (self.ori, self.dest)

    def get_vehicle(self):
        return self.vehicle

    def arrive(self, v):
        leave_time = self.time + int(self.dist/v.get_velocity())
        self.vehicle.append( (v, leave_time) )
        logging.info('Time {}: {} arrive at road ({},{})'.format(self.time, v.get_id(), self.ori, self.dest))

    def syn_time(self, time):
        self.time = time

    def leave(self, g):
        # leave_vehicle = []
        for (v, leave_time) in self.vehicle:
            # (v, leave_time) = self.vehicle[v_index]
            if (leave_time == self.time):
                self.vehicle.remove( (v, leave_time) )
                # leave_vehicle.append(v)
                
                logging.info('Time {}: {} leave road ({},{})'.format(self.time, v.get_id(), self.ori, self.dest))
                g.get_graph_dic()[self.dest]['node'].vehicle_arrive(v)


    
