import os
import logging

from SimMultiTrans.bin import Simulator
from SimMultiTrans.bin.Network import Graph
import SimMultiTrans.utils as utils

__all__ = [
    "Simulator",
    "Graph",
    "graph_file",
    "vehicle_file",
    "REBALANCE_POLICY",
    "ROUTING_POLICY",
    "utils",
]

os.makedirs(utils.RESULTS, exist_ok=True)
logging.basicConfig(level=logging.WARNING, filename=os.path.join(utils.RESULTS, 'Simulator.log'))

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
