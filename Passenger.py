#!/usr/bin/env python
# -*- coding: utf-8 -*-

class Passenger(object):
    def __init__(self, pid, ori, dest, arr_time):
        self.id = pid
        self.ori = ori
        self.dest = dest
        self.arr_time = arr_time

        self.loc = self.ori
        self.mode_wait = 0

        self.path = []

    def get_id(self):
        return self.id
    
    def get_odpair(self):
        return (self.ori, self.dest)

    def get_schdule(self, graph):
        self.path = graph.get_path(self.ori, self.dest)
        # print(self.path)
        return self.path

    def set_location(self, loc):
        self.loc = loc

    def get_waitingmode(self, loc):
        return self.path[loc]['info']['mode']

    def geton(self, loc, v):
        # print(self.path)
        if ( loc in self.path and self.path[loc]['info']['mode'] == v.get_mode()):
            # get on the correct vehicle (orientation)
            v.match_route(loc, self.path[loc]['dest'])
            return True
        return False
        # return True if next((edge for edge in self.path if (edge[0] == loc, edge[2]['mode'] == mode) ), False) else False

    def getoff(self, loc):
        # return True if next((edge for edge in self.path if edge[1] == loc), False) else False
        for node in self.path:
            if (self.path[node]['dest'] == loc):
                del(self.path[node])
                return True
        return False

    def get_nextstop(self, loc):
        return self.path[loc]['dest']





        