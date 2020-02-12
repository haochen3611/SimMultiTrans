from Graph import Graph
from Passenger import Passenger


import numpy as np
import random


def main():
    g = Graph()
    # print(g.get_allnodes())

    # Generate random passengers
    for i in range(1,100):
        
        nodes_set = g.get_allnodes()
        p_ori = random.choice(nodes_set)
        print(p_ori)
        nodes_set.remove(p_ori)
        # print(nodes_set)
        p_dest = random.choice(nodes_set)
        print(p_dest)
        
        p = Passenger(id=0, ori=p_ori, dest=p_dest, arr_time=0)
        # p = Passenger(id=0, ori='A1', dest='C3', arr_time=0)
        print(p.get_schdule(graph=g, time=0))
        # print(p.get_mode_wait(pos='A'))
        '''
        p = Passenger(id=0, ori='A', dest='C1', arr_time=0)
        print(p.get_schdule(graph=g, time=0))
        # print(p.get_mode_wait(pos='A'))
        '''
    
    


if __name__ == "__main__":
    main()