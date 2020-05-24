import os
import logging

from SimMultiTrans.bin import Simulator, Plot
from SimMultiTrans.bin.Network import Graph
import SimMultiTrans.utils as utils
from SimMultiTrans.dev import SimpleSimulator

__all__ = [
    "Simulator",
    "Graph",
    "Plot",
    "graph_file",
    "vehicle_file",
    "REBALANCE_POLICY",
    "ROUTING_POLICY",
    "utils",
    "default_graph",
    "SimpleSimulator"
]

os.makedirs(utils.RESULTS, exist_ok=True)
logging.basicConfig(level=logging.WARNING,
                    filename=os.path.join(utils.RESULTS, 'Simulator.log'),
                    filemode='w',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

if not os.path.exists(utils.CONFIG):
    logger.warning(f'{utils.CONFIG} not exist')
    os.makedirs(utils.CONFIG, exist_ok=True)

graph_file = 'city_nyc.json'
vehicle_file = 'vehicle_nyc.json'
REBALANCE_POLICY = 'Simplified_MaxWeight'
ROUTING_POLICY = 'simplex'
LAZINESS = 0
NEIGHBORS = 20

graph_file = os.path.join(utils.CONFIG, graph_file)
vehicle_file = os.path.join(utils.CONFIG, vehicle_file)


def default_graph():
    g = Graph()
    g.import_graph(graph_file)
    return g
