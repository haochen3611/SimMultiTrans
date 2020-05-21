import numpy as np
import numba as nb
import itertools
from collections import deque
import timeit


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

print('dfaf'
      'dsaf')
