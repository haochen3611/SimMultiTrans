#!/usr/bin/env python
# -*- coding: utf-8 -*-

class Passenger(object):
    def __init__(self, id, ori, dest, arr_time):
        self.id = id
        self.ori = ori
        self.dest = dest
        self.arr_time = arr_time

        self.cur_pos = self.ori
        self.mode_wait = 0

        self.path = []

    def get_schdule(self, graph, time):
        self.path = graph.get_path(self.ori, self.dest)
        # print(self.path)
        return self.path

    def get_mode_wait(self, pos):
        for p in self.path:
            if (p[0] == pos):
                return p[1][1]


        