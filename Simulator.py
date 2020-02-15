#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Graph import Graph
from Passenger import Passenger
from Vehicle import Vehicle
from Node import Node

import numpy as np
import random
import json

class Simulator(object):
    def __init__(self):
        self.graph = {}
        self.time_horizon = 0
        self.time = 0
        self.passenger = []

    def create_simulator(self, graph):
        self.graph = graph
        self.graph.generate_nodes()

    def set_time_horizon(self, th): 
        self.time_horizon = th

    def set_running_time(self, th, unit):
        unit_trans = {
            'day': 60*60*24,
            'hour': 60*60,
            'min': 60,
            'sec': 1
        }
        self.time_horizon = th*unit_trans[unit]

    def ori_dest_generator(self, method):
        if ( method.equal('uniform') ):
            # Generate random passengers
            nodes_set = self.graph.get_allnodes()
            ori = random.choice(nodes_set)
            # print('ori: ',p_ori)
            nodes_set.remove(ori)
            #print(nodes_set)
            dest = random.choice(nodes_set)            
            return (ori, dest)

    '''
    def import_passenger_attribute(self, file_name):
        with open('{}'.format(file_name)) as file_data:
            json_data = json.load(file_data)

    def passenger_generator(self, loc_d, time):
        (ori, dest) = self.ori_dest_generator(loc_d)

        p = Passenger(pid=0, ori=ori, dest=dest, arr_time=time)
        p.get_schdule(graph=self.graph)

    def import_vehicle_attribute(self, file_name):
        with open('{}'.format(file_name)) as file_data:
            json_data = json.load(file_data)

    def vehicle_generator(self):
        v = Vehicle(vid=0, mode='scooter', ori='A')
    
    '''    


