import numpy as np
import numba as nb
import itertools
from collections import deque
import timeit
import unittest


TEST = int(1e6)
LONG_LIST = [1]*int(1e6)


@nb.jit()
def for_loop_range(x):
    tt = 0
    for _ in nb.prange(x):
        tt += 1


def for_loop_repeat(x):
    tt = 0
    for _ in itertools.repeat(None, x):
        tt += 1


def enq_deq(x):
    d = deque([1, 2, 3])
    for _ in range(x):
        p = d.pop()
        d.append(p)


@nb.jit()
def enq_deq_list(x, d):
    for _ in range(x):
        p = d.pop(0)
        d.append(p)


# @nb.jit(nopython=True, parallel=True, fastmath=True)
def match_storage_demand(storage, demand, actual_trans):
    mat_sum = demand.sum(axis=1)
    has_more = np.greater_equal(storage, mat_sum)
    actual_trans[has_more, :] += demand[has_more, :]

    ava = storage[~has_more]
    need = demand[~has_more, :]
    for idx in nb.prange(len(ava)):
        for jdx in nb.prange(len(need[idx])):
            while need[idx][jdx] > 0 and ava[idx] > 0:
                temp = actual_trans[~has_more]
                temp[idx][jdx] += 1
                actual_trans[~has_more] = temp
                need[idx][jdx] -= 1
                ava[idx] -= 1
    return has_more, ava, need, mat_sum, actual_trans


class MSDTest(unittest.TestCase):

    def setUp(self) -> None:
        self.rng = np.random.default_rng()
        self.test_amount = 100
        self.size = 100
        self.upper = 10

    def test_msd(self):
        for i in range(self.test_amount):
            we_have = self.rng.integers(0, self.upper, size=(self.size,))
            they_want = self.rng.integers(0, self.upper, size=(self.size, self.size))
            trans = np.zeros((self.size, self.size), dtype=np.int)
            a, b, c, d, e = match_storage_demand(we_have, they_want, trans)
            with self.subTest(i=i):
                self.assertEqual(np.linalg.norm(e+c-they_want), 0)
                self.assertEqual(np.linalg.norm(b+e.sum(axis=1)-we_have), 0)


if __name__ == '__main__':
    unittest.main()
