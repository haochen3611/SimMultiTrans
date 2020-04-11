# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import logging


class Road(object):
    def __init__(self, ori, dest, dist=0, time=0):
        self.ori = ori
        self.dest = dest
        self.dist = dist
        self.time = 0
        self.triptime = time

        self.vehicle = []
        self.v_count = {}
        self.v_reb_count = {}

    def arrive(self, v):
        time = self.triptime
        if time == 0 or not v.onroad:
            time = int(self.dist / v.get_velocity('m/s'))
        leave_time = self.time + time
        self.vehicle.append((v, leave_time))
        logging.info(f'Time {self.time}: Vel {v.id} arrive at road ({self.ori},{self.dest})')

        if v.mode in self.v_count:
            self.v_count[v.mode] += 1
            if v.mode in self.v_reb_count and v.reb == 'active' and v.get_occupiedseats() == 0:
                self.v_reb_count[v.mode] += 1
        else:
            self.v_count[v.mode] = 1
            if v.mode not in self.v_reb_count and v.reb == 'active' and v.get_occupiedseats() == 0:
                self.v_reb_count[v.mode] = 1

    def leave(self, g):
        # leave_vehicle = []
        for (v, leave_time) in self.vehicle:
            # (v, leave_time) = self.vehicle[v_index]
            if leave_time == self.time:
                self.vehicle.remove((v, leave_time))
                # leave_vehicle.append(v)

                logging.info(f'Time {self.time}: Vel {v.id} leave road ({self.ori},{self.dest})')
                g.graph_top[self.dest]['node'].vehicle_arrive(v)

    def get_flow(self):
        return len(self.vehicle)

    def get_total_trip(self, mode):
        return 0 if (mode not in self.v_count) else self.v_count[mode]

    def get_total_reb_trip(self, mode):
        return 0 if (mode not in self.v_reb_count) else self.v_reb_count[mode]

    def get_total_time(self, mode):
        return 0 if (mode not in self.v_count) else self.triptime * self.v_count[mode]

    def get_total_reb_time(self, mode):
        return 0 if (mode not in self.v_reb_count) else self.triptime * self.v_reb_count[mode]

    def get_total_distance(self, mode):
        return 0 if (mode not in self.v_count) else self.dist * self.v_count[mode]

    def get_total_reb_distance(self, mode):
        return 0 if (mode not in self.v_reb_count) else self.dist * self.v_reb_count[mode]
