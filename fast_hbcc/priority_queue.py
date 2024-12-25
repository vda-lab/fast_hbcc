import numba as nb
from numba.types import ListType, NamedTuple
from heapq import heappush, heappop
from collections import namedtuple

# TODO: Using a Fibonacci heap would be better, but numba does not support
# self-referential types without going through a c struct api...

Node = namedtuple("Node", ["distance", "child", "depth"])
NumbaNode = NamedTuple([nb.float32, nb.int32, nb.int32], Node)
NumbaQueue = ListType(NumbaNode)


@nb.njit(NumbaQueue())
def make_queue():
    return nb.typed.List.empty_list(NumbaNode)


@nb.njit((NumbaQueue, nb.float32, nb.int32, nb.int32))
def push(queue, distance, child, depth):
    node = Node(distance, child, depth)
    heappush(queue, node)


@nb.njit(NumbaNode(NumbaQueue))
def pop(queue):
    return heappop(queue)


@nb.njit((NumbaQueue, nb.float32, nb.int32, nb.int32))
def append(stack, distance, child, depth):
    stack.append(Node(distance, child, depth))


@nb.njit(NumbaNode(NumbaQueue))
def pop_last(stack):
    return stack.pop()
