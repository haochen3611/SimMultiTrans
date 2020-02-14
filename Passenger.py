#!/usr/bin/env python
# -*- coding: utf-8 -*-

class Passenger(object):
    def __init__(self, pid, ori, dest, arr_time):
        self.id = pid
        self.ori = ori
        self.dest = dest
        self.arr_time = arr_time

        self.cur_loc = self.ori
        self.mode_wait = 0

        self.path = []

    def get_schdule(self, graph):
        self.path = graph.get_path(self.ori, self.dest)
        # print(self.path)
        return self.path

    def set_location(self, loc):
        self.cur_loc = loc

    def getoff(self, pos):
        # print(self.path)
        res = next((e for e in self.path if e[1] == pos), False)
        # print(res)
        return True if res else False





        