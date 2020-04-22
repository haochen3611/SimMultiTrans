import os
import logging

from SimMultiTrans.bin import *
from SimMultiTrans.bin.Control import *
from SimMultiTrans.bin.Network import *


__all__ = [
    "Simulator",
    "Graph",
    "graph_file",
    "vehicle_file",
    "REBALANCE_POLICY",
    "ROUTING_POLICY",
    "RESULTS",
    "CONFIG",
    "Plot"
]

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(ROOT, 'conf')
RESULTS = os.path.join(ROOT, 'results')

os.makedirs(RESULTS, exist_ok=True)
logging.basicConfig(level=logging.WARNING, filename=os.path.join(RESULTS, 'Simulator.log'))

logger = logging.getLogger(__name__)

if not os.path.exists(CONFIG):
    logger.warning(f'{CONFIG} not exist')
    os.makedirs(CONFIG, exist_ok=True)

graph_file = 'city_nyc.json'
vehicle_file = 'vehicle_nyc.json'
REBALANCE_POLICY = 'Simplified_MaxWeight'
ROUTING_POLICY = 'simplex'
LAZINESS = 0
NEIGHBORS = 20

graph_file = os.path.join(CONFIG, graph_file)
vehicle_file = os.path.join(CONFIG, vehicle_file)
