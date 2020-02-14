from Graph import Graph
from Passenger import Passenger


import numpy as np
import random


def main():
    g = Graph('city.json')
    print(g.get_edge('A1','A'))

    # Generate random passengers
    nodes_set = g.get_allnodes()
    p_ori = random.choice(nodes_set)
    print('ori: ',p_ori)
    nodes_set.remove(p_ori)
    #print(nodes_set)
    p_dest = random.choice(nodes_set)
    print('dest: ',p_dest)
    
    # p = Passenger(id=0, ori=p_ori, dest=p_dest, arr_time=0)
    p = Passenger(id=0, ori='A1', dest='C1', arr_time=0)
    print(p.get_schdule(graph=g, time=0))
    
    g.generate_nodes()
    


if __name__ == "__main__":
    main()