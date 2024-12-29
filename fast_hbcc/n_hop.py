"""
Parallel depth-limit traversal for Dijkstra's all-pair shortest path and
depth-first traversal. Numba limits the design of this code. When jitclass
becomes stable, the _path functions can be refactored to reduce duplication.
"""

import numba as nb
import numpy as np
from typing import Literal
from fast_hdbscan.numba_kdtree import rdist
from fast_hdbscan.core_graph import knn_mst_union

from .priority_queue import make_queue, push, append, pop, pop_last


@nb.njit(parallel=True)
def to_csr(children):
    num_points = len(children)
    indptr = np.empty(num_points + 1, dtype=np.int32)
    indptr[0] = 0
    for i, c in enumerate(children):
        indptr[i + 1] = indptr[i] + len(c)

    data = np.empty(indptr[-1], dtype=np.float32)
    indices = np.empty(indptr[-1], dtype=np.int32)
    for point in nb.prange(num_points):
        idx = indptr[point]
        for count, (child, dist) in enumerate(children[point].items()):
            indices[idx + count] = child
            data[idx + count] = dist

    return (data, indices, indptr)


# --- Manifold n-hop ---


@nb.njit()
def manifold_path(root, graph, max_depth=2):
    """Dijkstra depth-limited traversal."""
    queue = make_queue()
    children = {np.int32(root): np.float32(0.0) for _ in range(0)}
    for child, dist in graph[root].items():
        push(queue, dist, child, np.int32(1))
    while queue:
        distance, point, depth = pop(queue)
        if point in children:
            continue
        children[point] = distance
        if depth == max_depth:
            continue
        for child, dist in graph[point].items():
            if child == root or child in children:
                continue
            push(queue, distance + dist, child, depth + 1)
    return children


@nb.njit(
    parallel=True, locals={"root": nb.int32, "child": nb.int32, "dist": nb.float32}
)
def manifold_knn_hop(distances, indices, max_depth=2):
    result = [
        {np.int32(0): np.float32(0.0) for _ in range(0)}
        for _ in range(indices.shape[0])
    ]

    for root in range(indices.shape[0]):
        for child, dist in zip(indices[root], distances[root]):
            if child < 0:
                continue
            result[root][child] = dist
            result[child][root] = dist

    for root in nb.prange(indices.shape[0]):
        result[root] = manifold_path(root, result, max_depth=max_depth)

    return to_csr(result)


@nb.njit(parallel=True, locals={"root": nb.int32, "child": nb.int32})
def manifold_mst_hop(
    data,
    indices,
    core_distances,
    minimum_spanning_tree,
    max_depth=2,
    use_reachability=True,
):
    num_points = indices.shape[0]
    core_graph = knn_mst_union(
        indices, core_distances, minimum_spanning_tree, core_distances
    )
    if not use_reachability:
        for point in nb.prange(num_points):
            for child, value in core_graph[point].items():
                core_graph[point][child] = (
                    value[0],
                    np.sqrt(rdist(data[point], data[child])),
                )

    result = [
        {np.int32(0): np.float32(0.0) for _ in range(0)} for _ in range(num_points)
    ]

    for root in range(num_points):
        for child, dist in core_graph[root].items():
            result[root][child] = np.float32(dist[1])
            result[child][root] = np.float32(dist[1])

    for root in nb.prange(num_points):
        result[root] = manifold_path(root, result, max_depth=max_depth)
    return to_csr(result)


def manifold_n_hop(
    data,
    indices,
    core_distances,
    minimum_spanning_tree,
    max_depth=2,
    connectivity="knn",
    use_reachability=True,
):
    if connectivity == "knn":
        if use_reachability:
            distances = np.maximum(core_distances[indices], core_distances[:, None])
        else:
            distances = np.linalg.norm(data[indices] - data[:, None], axis=-1)
        return manifold_knn_hop(distances, indices, max_depth=max_depth)
    return manifold_mst_hop(
        data,
        indices,
        core_distances,
        minimum_spanning_tree,
        max_depth=max_depth,
        use_reachability=use_reachability,
    )


# --- Metric n-hop ---


@nb.njit()
def metric_path(root, data, graph, core_distances, max_depth=2, use_reachability=True):
    """Depth first depth-limit traversal."""
    stack = make_queue()
    enqueued = {root}
    for child in graph[root].keys():
        distance = np.sqrt(rdist(data[root], data[child]))
        if use_reachability:
            distance = max(distance, core_distances[root], core_distances[child])
        append(stack, distance, child, np.int32(1))
        enqueued.add(child)

    children = {np.int32(root): np.float32(0.0) for _ in range(0)}
    while stack:
        distance, point, depth = pop_last(stack)
        children[point] = distance
        if depth == max_depth:
            continue
        for child in graph[point].keys():
            if child in enqueued:
                continue
            distance = np.sqrt(rdist(data[root], data[child]))
            if use_reachability:
                distance = max(distance, core_distances[root], core_distances[child])
            push(stack, distance, child, depth + 1)
            enqueued.add(child)
    return children


@nb.njit(parallel=True, locals={"root": nb.int32, "child": nb.int32})
def metric_knn_hop(
    data,
    indices,
    core_distances,
    max_depth=2,
    use_reachability=True,
):
    result = [
        {np.int32(0): np.float32(0.0) for _ in range(0)}
        for _ in range(indices.shape[0])
    ]

    for root in range(indices.shape[0]):
        for child in indices[root]:
            if child < 0:
                continue
            result[root][child] = np.float32(0.0)
            result[child][root] = np.float32(0.0)

    for root in nb.prange(indices.shape[0]):
        result[root] = metric_path(
            root,
            data,
            result,
            core_distances,
            max_depth=max_depth,
            use_reachability=use_reachability,
        )
    return to_csr(result)


@nb.njit(parallel=True, locals={"root": nb.int32, "child": nb.int32})
def metric_mst_hop(
    data,
    indices,
    core_distances,
    minimum_spanning_tree,
    max_depth=2,
    use_reachability=True,
):
    core_graph = knn_mst_union(
        indices, core_distances, minimum_spanning_tree, core_distances
    )

    result = [
        {np.int32(0): np.float32(0.0) for _ in range(0)}
        for _ in range(indices.shape[0])
    ]

    for root in range(indices.shape[0]):
        for child in core_graph[root].keys():
            result[root][child] = np.float32(0.0)
            result[child][root] = np.float32(0.0)

    for root in nb.prange(indices.shape[0]):
        result[root] = metric_path(
            root,
            data,
            result,
            core_distances,
            max_depth=max_depth,
            use_reachability=use_reachability,
        )
    return to_csr(result)


def metric_n_hop(
    data,
    indices,
    core_distances,
    minimum_spanning_tree,
    max_depth=2,
    connectivity="knn",
    use_reachability=True,
):
    if connectivity == "knn":
        return metric_knn_hop(
            data,
            indices,
            core_distances,
            max_depth=max_depth,
            use_reachability=use_reachability,
        )
    return metric_mst_hop(
        data,
        indices,
        core_distances,
        minimum_spanning_tree,
        max_depth=max_depth,
        use_reachability=use_reachability,
    )
