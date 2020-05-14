#!/usr/bin/env python
# -*- coding: utf-8 -*-


class Passenger(object):
    def __init__(self, pid, ori, dest, arr_time):
        self.id = pid
        self.ori = ori
        self.dest = dest
        self.arr_time = arr_time
        self.stop_time = arr_time

        self.loc = self.ori
        self.mode_wait = 0

        self.path = []

    def get_schdule(self, routing):
        self.path = routing.get_path(self.ori, self.dest)

        return self.path

    def get_waitingmode(self, loc):

        return self.path[loc]['info']['mode'] if loc in self.path and loc != self.dest else None

    def geton(self, loc, v):
        # print(self.path)
        """
        if ( loc in self.path and self.path[loc]['info']['mode'] == v.mode):
            # get on the correct vehicle (orientation)

            return v.match_route(loc, self.path[loc]['dest'])
        return False
        """
        return v.match_route(loc, self.path[loc]['dest']) if loc in self.path and self.path[loc]['info'][
            'mode'] == v.mode else False

    def getoff(self, loc):
        # return True if next((edge for edge in self.path if edge[1] == loc), False) else False
        for node in self.path:
            if self.path[node]['dest'] == loc:
                del (self.path[node])
                return True
        return False

    def get_nextstop(self, loc):
        return self.path[loc]['dest'] if (loc in self.path and loc != self.dest) else None




