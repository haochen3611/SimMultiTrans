
#!/usr/bin/env python
# -*- coding: utf-8 -*-

class Node(object):
    def __init__(self, nid, loc, mode):
        self.id = nid
        self.loc = loc
        self.mode = mode

        # print(self.id, self.loc, self.mode)
        self.passenger = []
        self.vehicle = {}

        for m in self.mode:
            self.vehicle[m] = {}
    
    def get_id(self):
        return self.id

    def get_location(self):
        return self.loc

    def get_mode(self):
        return self.mode

    def check_accessiblity(self, mode):
        return (mode in self.mode)

    def passenger_arrival(self, p):
        self.passenger.append(p)

    def passenger_leave(self, p):
        self.passenger.remove(p)