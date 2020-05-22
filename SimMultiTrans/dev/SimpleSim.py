import numpy as np
import json
from collections import deque
import heapq
import numba as nb
import matplotlib.pyplot as plt
import pandas as pd
import line_profiler as lpf
import time

from SimMultiTrans.bin.Network.Node import Haversine
from SimMultiTrans import graph_file, vehicle_file
from SimMultiTrans.utils import update_graph_file, update_vehicle_initial_distribution, CONFIG


class BaseSimulator:
    """Base class of simulator should never be called directly"""

    def __init__(self):
        """
        Implement constructor in subclass
        """

    def _initialize_from_configs(self, *args, **kwargs):
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

    def _passenger_arrival(self, *args, **kwargs):
        """
        Only consider new passenger arrival. Does not count arriving passengers from other nodes.
        Add new passenger matrix to the passenger queue at every second.
        :param args:
        :param kwargs:
        :return:
        """
        pass

    def _passenger_leave(self, *args, **kwargs):
        """
        Happens simultaneously with vehicle_leave
        :param args:
        :param kwargs:
        :return:
        """
        pass

    def _vehicle_arrival(self, *args, **kwargs):
        """
        Add the column sum of vehicle transition matrix to vehicle queue
        :param args:
        :param kwargs:
        :return:
        """
        pass

    def _vehicle_leave(self, *args, **kwargs):
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
    """
    Not yet integrate numba acceleration
    """

    def __init__(self, graph_config, vehicle_config, time_horizon, time_unit='hour', seed=0):
        super().__init__()

        # private
        self._time_horizon = self._time_unit_converter(time_horizon, time_unit)
        # self._original_time = time_horizon
        # self._time_unit = time_unit

        self._g_file = graph_config
        self._v_file = vehicle_config
        self._seed = seed
        self._vehicle_queue = np.zeros(0)
        # self._passenger_queue = deque(maxlen=self._time_horizon)  # cannot use with numba
        self._passenger_queue = list()
        self._node_pass_sum = np.zeros(0)  # Consider caching the queue sum for speed
        self._travel_time = np.zeros(0)
        self._travel_dist = np.zeros(0)
        self._arr_rate = np.zeros(0)
        self._passenger_schedule = np.zeros(0)
        # self._vehicle_schedule = deque(maxlen=self._time_horizon)
        self._vehicle_schedule = list()
        self._cur_time = 0
        self._rdn = np.zeros(0)
        self._rng = np.random.default_rng(self._seed)

        # public
        self.num_nodes = 0
        self.avg_veh_speed = 0
        self.node_to_index = dict()
        self.index_to_node = list()
        self.v_q_history = list()
        self.p_q_history = list()
        self.reb_history = list()
        self.imbalance = list()

        with open(self._g_file, 'r') as g_file:
            self.g_config = json.load(g_file)
        with open(self._v_file, 'r') as v_file:
            self.v_config = json.load(v_file)

        # initialize everything except for passengers
        # self._initialize_from_configs(self.g_config, self.v_config)
        # initialize passengers
        # self._generate_passengers()

    @property
    def veh_queue(self):
        return self._vehicle_queue.copy()

    @property
    def pass_queue(self):
        pass_queue_sum = np.zeros(self.num_nodes, dtype=np.int64)
        for pass_queue in self._passenger_queue:
            pass_queue_sum += pass_queue.sum(axis=1)
        return pass_queue_sum

    @staticmethod
    def _time_unit_converter(time_, unit):
        assert isinstance(unit, str)
        unit = unit.lower()
        if unit in ['hour', 'h']:
            return int(time_ * 3600)
        elif unit in ['min', 'm', 'minute']:
            return int(time_ * 60)
        elif unit in ['second', 's', 'sec']:
            return int(time_)
        else:
            raise ValueError(f"Invalid value received for \'unit\': {unit}")

    def _initialize_from_configs(self, graph_config, vehicle_config):
        self.num_nodes = len(graph_config)
        self.avg_veh_speed = vehicle_config['taxi']['vel']

        self.node_to_index = dict()
        for idx, nd in enumerate(graph_config):
            self.node_to_index[nd] = idx
        self.index_to_node = list(graph_config.keys())

        # initialize travel time and distance
        self._travel_time = np.zeros((self.num_nodes, self.num_nodes), dtype=np.int64)
        self._travel_dist = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float64)
        self._arr_rate = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float64)
        for start in graph_config:
            for end in graph_config[start]['nei']:
                l1_dist = Haversine((graph_config[start]['lat'], graph_config[start]['lon']),
                                    (graph_config[end]['lat'], graph_config[end]['lon'])).meters
                # travel time from one node to itself is 1 second
                if ('time' not in graph_config[start]['nei'][end]) or (graph_config[start]['nei'][end]['time'] == 0):
                    self._travel_time[self.node_to_index[start],
                                      self.node_to_index[end]] = np.int64(np.ceil(l1_dist / self.avg_veh_speed)) \
                        if start != end else 1
                else:
                    self._travel_time[self.node_to_index[start],
                                      self.node_to_index[end]] = graph_config[start]['nei'][end]['time']
                # update graph config
                graph_config[start]['nei'][end]['dist'] = l1_dist
                # travel distance
                self._travel_dist[self.node_to_index[start],
                                  self.node_to_index[end]] = l1_dist
                # initialize arrival rate
                self._arr_rate[self.node_to_index[start],
                               self.node_to_index[end]] = graph_config[start]['nei'][end]['rate'] if start != end else 0
        # Over rewrite graph config
        self.g_config = graph_config
        # initialize vehicles
        self._vehicle_queue = np.zeros(self.num_nodes, dtype=np.int64)
        for nd in vehicle_config["taxi"]["distrib"]:
            self._vehicle_queue[self.node_to_index[nd]] = int(vehicle_config["taxi"]["distrib"][nd])

        # initialize passengers
        # use a separate function due numba compatibility issue

    def _generate_passengers(self, arr_rate=None, time_horizon=None):
        """
        The total passengers generated are not the same as the original simulator
        :param time_horizon:
        :return:
        """
        if arr_rate is None:
            arr_rate = self._arr_rate
        else:
            assert isinstance(arr_rate, (int, np.ndarray, list, tuple))
            if isinstance(arr_rate, (list, tuple)):
                arr_rate = np.array(arr_rate)
        if time_horizon is None:
            time_horizon = self._time_horizon
        else:
            assert isinstance(time_horizon, int)

        self._rdn = self._rng.uniform(0, 1, size=(time_horizon, self.num_nodes, self.num_nodes))
        arr_prob = 1 - np.exp(-arr_rate)
        self._passenger_schedule = np.greater(arr_prob, self._rdn).astype(np.int64)
        # self._passenger_schedule = pass_gen(arr_rate=arr_rate,
        #                                     time_horizon=time_horizon,
        #                                     size=self.num_nodes)

    def _passenger_arrival(self):
        """Call every second
        Will not do checking for speed
        """
        if np.sum(self._passenger_schedule[self._cur_time]) != 0:
            self._passenger_queue.append(self._passenger_schedule[self._cur_time].copy())  # mutable
            # print(f'time {self._cur_time} pass queue length {len(self._passenger_queue)}')

    @profile
    def _match_demand_and_dispatch(self, dispatch_mat=None):
        """Call every second
        Match passenger demand and dispatch empty vehicles
        Handles vehicle leave and passenger leave. No need for separate functions
        Assume no passenger from one node to itself (forced arrival rate to be zero in configuration)
        :param dispatch_mat: must be ndarray of np.int64 type
        :return:
        """
        veh_trans_mat = np.zeros((self.num_nodes, self.num_nodes), dtype=np.int64)

        start = -1
        for time_pt in range(len(self._passenger_queue)):
            # TODO: Need something like a double-pointer to better trace passenger queue
            cur_p_q = self._passenger_queue[time_pt]
            # cur_q_sum = cur_p_q.sum(axis=1)
            # has_more_veh = np.greater_equal(self._vehicle_queue, cur_q_sum)

            has_more_veh, extra_veh, unmatched_pass, cur_q_sum, veh_trans_mat = \
                match_storage_demand(storage=self._vehicle_queue,
                                     demand=cur_p_q,
                                     actual_trans=veh_trans_mat)

            # nodes with more vehicles than passengers
            self._vehicle_queue[has_more_veh] -= cur_q_sum[has_more_veh]
            self._passenger_queue[time_pt][has_more_veh, :] -= cur_p_q[has_more_veh, :]
            # nodes with less vehicles than passengers
            self._vehicle_queue[~has_more_veh] = extra_veh  # should be zero
            # print(f'time {self._cur_time} extra_veh {extra_veh}')
            if np.sum(unmatched_pass) != 0:
                self._passenger_queue[time_pt][~has_more_veh, :] = unmatched_pass  # if passenger left put back on queue
                # print(f'time {self._cur_time} extra pass{unmatched_pass}')
            else:
                start = time_pt + 1

        # get rid of zero queues
        if start >= 0:
            self._passenger_queue = self._passenger_queue[start:]

        # put vehicle schedule on heap queue
        # np.where, np.nonzero, np.argwhere
        if dispatch_mat is not None:
            has_more_veh, extra_veh, unmatched, col_sum, dispatch_mat = \
                match_storage_demand(storage=self._vehicle_queue,
                                     demand=dispatch_mat,
                                     actual_trans=np.zeros(shape=dispatch_mat.shape, dtype=np.int64))
            self._vehicle_queue[has_more_veh] -= col_sum[has_more_veh]
            self._vehicle_queue[~has_more_veh] = extra_veh
            veh_trans_mat += dispatch_mat
        #     self.reb_history.append(dispatch_mat.sum(axis=1))
        # else:
        #     self.reb_history.append(np.zeros(self.num_nodes))
        for non_z in np.argwhere(veh_trans_mat):
            heapq.heappush(self._vehicle_schedule,
                           (self._travel_time[non_z[0], non_z[1]] + self._cur_time,
                            veh_trans_mat[non_z[0], non_z[1]],
                            non_z[1]))

    def _vehicle_arrival(self):
        """Call every second
        Check the top of the heap to see if the scheduled event happens now
        Adds the arriving vehicles to the vehicle queue
        :return:
        """
        while True:
            try:
                if self._vehicle_schedule[0][0] == self._cur_time:
                    cur_arr_veh = heapq.heappop(self._vehicle_schedule)
                    self._vehicle_queue[cur_arr_veh[2]] += cur_arr_veh[1]
                    # print(f'time {self._cur_time} vehicle schedule {len(self._vehicle_schedule)}')
                    # print(f'time {self._cur_time} vehicle arrive {cur_arr_veh[1]} at node {cur_arr_veh[2]}')
                elif self._vehicle_schedule[0][0] < self._cur_time:
                    raise Exception(f'Vehicle queue fall behind. '
                                    f'Queue time: {self._vehicle_schedule[0][0]}. Current time: {self._cur_time}')
                else:
                    break
            except IndexError:
                break

    def _save_history(self):
        self.v_q_history.append(self.veh_queue)
        self.p_q_history.append(self.pass_queue)

    # @profile
    def _sim_one_second_routine(self, action=None):
        self._passenger_arrival()
        self._vehicle_arrival()
        # self.v_q_history.append(self.veh_queue)
        # self.p_q_history.append(self.pass_queue)
        self.imbalance.append(self.pass_queue-self.veh_queue)
        self._match_demand_and_dispatch(action)

    def step(self, action, step_length):
        """
        Input action must be a square matrix of probability with row sums equal to one.
        Will not check this assumption.
        :param action:
        :param step_length:
        :return:
        """
        assert isinstance(action, np.ndarray)
        assert action.shape == (self.num_nodes, self.num_nodes)

        veh_dispatch = np.rint(action * self._vehicle_queue[:, np.newaxis])
        for time_step in range(step_length):
            if (time_step+1) == step_length:
                self._sim_one_second_routine(action=np.int64(veh_dispatch))
            else:
                self._sim_one_second_routine()
            self._cur_time += 1
        return self.pass_queue, self.veh_queue

    def reset(self):
        """
        Need a more efficient one. But this will do for now
        :return:
        """
        self._vehicle_queue = np.zeros(0)
        # self._passenger_queue = deque(maxlen=self._time_horizon)  # cannot use with numba
        self._passenger_queue = list()
        self._node_pass_sum = np.zeros(0)  # Consider caching the queue sum for speed
        self._travel_time = np.zeros(0)
        self._travel_dist = np.zeros(0)
        self._arr_rate = np.zeros(0)
        self._passenger_schedule = np.zeros(0)
        # self._vehicle_schedule = deque(maxlen=self._time_horizon)
        self._vehicle_schedule = list()
        self._cur_time = 0
        self._rdn = np.zeros(0)
        self._rng = np.random.default_rng(self._seed)

        # public
        self.num_nodes = 0
        self.avg_veh_speed = 0
        self.node_to_index = dict()
        self.index_to_node = list()
        self.v_q_history = list()
        self.p_q_history = list()
        self.reb_history = list()

        # initialize everything except for passengers
        self._initialize_from_configs(self.g_config, self.v_config)
        # initialize passengers
        self._generate_passengers()

        return self.pass_queue, self.veh_queue

    def plot_results(self):

        # v_df = pd.DataFrame(self.v_q_history, columns=list(sim.node_to_index.keys()))
        # p_df = pd.DataFrame(self.p_q_history, columns=list(sim.node_to_index.keys()))
        # reb_df = pd.DataFrame(self.reb_history, columns=list(sim.node_to_index.keys()))
        imb_df = pd.DataFrame(self.imbalance, columns=list(sim.node_to_index.keys()))
        imb_df.plot(y=imb_df.columns, legend=True, title='System Imbalance')
        plt.savefig('/home/haochen/PycharmProjects/SimMultiTrans/SimMultiTrans/results/simlite-imbalance.png')
        # p_df.plot(y=p_df.columns, legend=True, title='Passenger Queue')
        # plt.show()
        # reb_df.plot(y=reb_df.columns, legend=True, title='Rebalanced Vehicles')
        # plt.show()


# @nb.jit(nopython=True, parallel=True, fastmath=True)
# @profile
def match_storage_demand(storage: np.ndarray, demand: np.ndarray, actual_trans: np.ndarray):
    """
    Separate for function for numba acceleration.
    Currently this is the slowest part of the entire sim
    May need improvement in the future
    :param storage: vector
    :param demand: square matrix
    :param actual_trans: square matrix. Modify the input.
    :return: has_more, need, mat_sum
    """
    mat_sum = demand.sum(axis=1)
    has_more = np.greater_equal(storage, mat_sum)

    actual_trans[has_more, :] += demand[has_more, :]

    ava = storage[~has_more]
    need = demand[~has_more, :]
    for idx in range(len(ava)):
        for jdx in range(len(need[idx])):
            while need[idx][jdx] > 0 and ava[idx] > 0:
                temp = actual_trans[~has_more]
                temp[idx][jdx] += 1
                actual_trans[~has_more] = temp
                need[idx][jdx] -= 1
                ava[idx] -= 1
    return has_more, ava, need, mat_sum, actual_trans


def pass_gen(arr_rate, time_horizon, size):
    """

    :param arr_rate:
    :param time_horizon:
    :param size:
    :return:
    """
    rd = np.random.uniform(0, 1, size=(time_horizon, size, size))
    arr_prob = 1 - np.exp(-arr_rate)
    toss = np.greater(arr_prob, rd).astype(np.int64)
    return toss


if __name__ == '__main__':

    time_hor = 10
    step_len = 600
    num_steps = time_hor * 3600 // step_len

    # NODES = sorted(pd.read_csv(os.path.join(CONFIG, 'aam.csv'), index_col=0, header=0).index.values.tolist())
    NODES = sorted([236, 237, 186, 170, 141, 162, 140, 238, 142, 229, 239, 48, 161, 107, 263, 262, 234, 68, 100, 143])
    # NODES = sorted([236, 237, 186, 170, 141])

    update_graph_file(NODES)
    update_vehicle_initial_distribution(nodes=NODES,
                                        veh_dist=[int(16) for i in range(len(NODES))])
    start_time = time.time()
    sim = SimpleSimulator(graph_config=graph_file,
                          vehicle_config=vehicle_file,
                          time_horizon=time_hor,
                          time_unit='hour')
    pq_lst = []
    vq_lst = []
    for step in range(num_steps + 1):
        if step == 0:
            pq, vq = sim.reset()
            print("Total Passengers:", np.sum(sim._passenger_schedule))
        else:
            # act = np.random.random((sim.num_nodes, sim.num_nodes))
            act = np.zeros((sim.num_nodes, sim.num_nodes))
            # act = np.eye(sim.num_nodes)
            pq, vq = sim.step(act, step_length=step_len)
        print(sim._cur_time)

    sim.plot_results()
    print("Time used:", time.time()-start_time)





















