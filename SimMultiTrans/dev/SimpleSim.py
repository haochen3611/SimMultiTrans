import numpy as np
import json
from collections import deque

from SimMultiTrans.bin.Network.Node import Haversine


class BaseSimulator:
    """Base class of simulator should never be called directly"""

    def __init__(self):
        """
        Implement constructor in subclass
        """

    def _read_config_files(self):
        """
        Read in graph file and parse the node info
        :return:
        """
        pass

    def reset(self):
        """
        Reset the entire graph to its initial state
        :return:
        """
        pass

    def _generate_passengers(self, *args, **kwargs):
        """
        Generate a three dimension matrix of new passenger from one node to another at every second
        :param args:
        :param kwargs:
        :return:
        """
        pass

    def passenger_arrival(self, *args, **kwargs):
        """
        Only consider new passenger arrival. Does not count arriving passengers from other nodes.
        Add new passenger matrix to the passenger queue at every second.
        :param args:
        :param kwargs:
        :return:
        """
        pass

    def passenger_leave(self, *args, **kwargs):
        """
        Happens simultaneously with vehicle_leave
        :param args:
        :param kwargs:
        :return:
        """
        pass

    def vehicle_arrival(self, *args, **kwargs):
        """
        Add the column sum of vehicle transition matrix to vehicle queue
        :param args:
        :param kwargs:
        :return:
        """
        pass

    def vehicle_leave(self, *args, **kwargs):
        """
        Subtract the row sum of vehicle transition matrix from the vehicle queue
        :param args:
        :param kwargs:
        :return:
        """
        pass

    def step(self, *args, **kwargs):
        """
        Call this function at each MDP step
        :param args:
        :param kwargs:
        :return:
        """
        pass

    def _get_vehicle_transition(self, *args, **kwargs):
        """
        Generate vehicle transition matrix. Find out how many vehicle with passengers need to be sent from nodes
        to nodes. Put their arrival in the event schedule. Then find out how many empty vehicle need to be dispatched
        from nodes to nodes and put their arrival in the event schedule.
        :param args:
        :param kwargs:
        :return:
        """
        pass


class SimpleSimulator(BaseSimulator):

    def __init__(self, graph_file, vehicle_file, time_horizon, time_unit='hour'):
        super().__init__()

        # private
        self._time_horizon = self._time_unit_converter(time_horizon, time_unit)
        self._g_file = graph_file
        self._v_file = vehicle_file
        self._vehicle_queue = np.zeros(0)
        self._passenger_queue = deque(maxlen=self._time_horizon)
        self._node_pass_sum = np.zeros(0)  # Consider caching the queue sum for speed
        self._travel_time = np.zeros(0)
        self._travel_dist = np.zeros(0)
        self._arr_rate = np.zeros(0)
        self._passenger_schedule = np.zeros(0)
        self._vehicle_schedule = deque(maxlen=self._time_horizon)
        self._cur_time = 0

        # public
        self.num_nodes = 0
        self.avg_veh_speed = 0
        self.node_to_index = dict()
        self.index_to_node = list()

        # initialize
        self._read_config_files()

    @property
    def veh_queue(self):
        return self._vehicle_queue.copy()

    @property
    def pass_queue(self):
        pass_queue_sum = np.zeros(self.num_nodes)
        for pass_queue in self._passenger_queue:
            pass_queue_sum += pass_queue.sum(axis=1)
        return pass_queue_sum

    @staticmethod
    def _time_unit_converter(time, unit):
        assert isinstance(unit, str)
        unit = unit.lower()
        if unit in ['hour', 'h']:
            return int(time * 3600)
        elif unit in ['min', 'm', 'minute']:
            return int(time * 60)
        elif unit in ['second', 's', 'sec']:
            return int(time)
        else:
            raise ValueError(f"Invalid value received for \'unit\': {unit}")

    def _read_config_files(self):
        with open(self._g_file, 'r') as g_file:
            graph_config = json.load(g_file)
        with open(self._v_file, 'r') as v_file:
            vehicle_config = json.load(v_file)
        self.num_nodes = len(graph_config)
        self.avg_veh_speed = vehicle_config['taxi']['vel']

        self.node_to_index = dict()
        for idx, nd in enumerate(graph_config):
            self.node_to_index[nd] = idx
        self.index_to_node = list(graph_config.keys())

        # initialize travel time and distance
        self._travel_time = np.zeros((self.num_nodes, self.num_nodes))
        self._travel_dist = np.zeros((self.num_nodes, self.num_nodes))
        for start in graph_config:
            for end in graph_config[start]['nei']:
                l1_dist = Haversine((graph_config[start]['lat'], graph_config[start]['lon']),
                                    (graph_config[end]['lat'], graph_config[end]['lon'].meters)).meters
                self._travel_time[self.node_to_index[start],
                                  self.node_to_index[end]] = l1_dist / self.avg_veh_speed \
                    if 'time' not in graph_config[start]['nei'][end] else graph_config[start]['nei'][end]['time']
                graph_config[start]['nei'][end]['dist'] = l1_dist
                self._travel_dist[self.node_to_index[start],
                                  self.node_to_index[end]] = l1_dist
        # initialize vehicles
        self._vehicle_queue = np.zeros(self.num_nodes)
        for nd in vehicle_config["taxi"]["distrib"]:
            self._vehicle_queue[self.node_to_index[nd]] = int(vehicle_config["taxi"]["distrib"][nd])

        # initialize passengers
        self._passenger_schedule = self._generate_passengers()

    def _exponential_dist(self, arr_rate=None, size=None):
        if arr_rate is None:
            arr_rate = self._arr_rate
        else:
            assert isinstance(arr_rate, (int, np.ndarray, list, tuple))
            if isinstance(arr_rate, (list, tuple)):
                arr_rate = np.array(arr_rate)
        if size is None:
            size = self.num_nodes
        else:
            assert isinstance(size, int)

        rd = np.random.uniform(0, 1, (size, size))
        rd /= rd.sum(axis=1)[:, np.newaxis]
        return 1 - np.exp(-arr_rate * rd)

    def _generate_passengers(self, time_horizon=None):
        """
        The total passengers generated are not the same as the original simulator
        :param time_horizon:
        :return:
        """
        if time_horizon is None:
            time_horizon = self._time_horizon
        else:
            assert isinstance(time_horizon, int)

        arr_prob = self._exponential_dist()
        rd = np.random.uniform(0, 1, size=(time_horizon, self.num_nodes, self.num_nodes))
        rd /= rd.sum(axis=2)[:, :, np.newaxis]
        new_pass = np.greater(arr_prob[np.newaxis, :, :], rd).astype(np.bool)
        return new_pass

    def passenger_arrival(self):
        """Call every second
        Will not do checking for speed
        """
        if np.sum(self._passenger_schedule[self._cur_time]) != 0:
            self._passenger_queue.appendleft(self._passenger_schedule[self._cur_time])

    def _match_demand(self):
        # curr_pass_queue = self.pass_queue
        # has_more_veh = np.greater(self._vehicle_queue, curr_pass_queue)
        veh_trans_mat = np.zeros((self.num_nodes, self.num_nodes))

        while self._passenger_queue:
            cur_p_q = self._passenger_queue.pop()
            cur_q_sum = cur_p_q.sum(axis=1)
            has_more_veh = np.greater_equal(self._vehicle_queue, cur_q_sum)

            # nodes with more vehicle than passenger
            self._vehicle_queue[has_more_veh] -= cur_q_sum[has_more_veh]
            veh_trans_mat[has_more_veh, :] += cur_p_q[has_more_veh, :]

            # nodes with less vehicle than passenger




















