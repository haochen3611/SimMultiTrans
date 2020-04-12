import os

from SimMultiTrans.bin import *
from SimMultiTrans.bin.Control import *
from SimMultiTrans.bin.Network import *
from SimMultiTrans.utils import CONFIG


__all__ = [
    "Simulator",
    "Graph",
    "graph_file",
    "vehicle_file",
    "p_name",
    "r_name",
]

graph_file = 'city_nyc.json'
vehicle_file = 'vehicle_nyc.json'
p_name = 'Simplified_MaxWeight'
r_name = 'simplex'
lazy = 0
radius = 20

graph_file = os.path.join(CONFIG, graph_file)
vehicle_file = os.path.join(CONFIG, vehicle_file)

# graph = Graph()
# graph.import_graph(graph_file)

# sim = Simulator(graph=graph)
# sim.import_arrival_rate(unit=(1, 'sec'))
# sim.import_vehicle_attribute(file_name=vehicle_file)
# sim.set_multiprocessing(False)
#
# sim.routing.set_routing_method(r_name)
# sim.rebalance.set_parameters(lazy=lazy, vrange=radius)
# sim.rebalance.set_policy(p_name)
