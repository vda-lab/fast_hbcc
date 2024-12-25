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
def manifold_knn_path(root, distances, indices, max_depth=2):
    """Dijkstra depth-limited traversal."""
    queue = make_queue()
    children = {np.int32(root): np.float32(0.0) for _ in range(0)}
    for dist, child in zip(distances[root], indices[root]):
        push(queue, dist, child, np.int32(1))
    while queue:
        distance, point, depth = pop(queue)
        if point in children:
            continue
        children[point] = distance
        if depth == max_depth:
            continue
        for dist, child in zip(distances[point], indices[point]):
            if child == root or child in children:
                continue
            push(queue, distance + dist, child, depth + 1)
    return children


@nb.njit()
def manifold_mst_path(root, core_graph, max_depth=2, use_reachability=True):
    """Dijkstra depth-limited traversal."""
    queue = make_queue()
    idx = 1 if use_reachability else 0
    children = {np.int32(root): np.float32(0.0) for _ in range(0)}
    for child, dist in core_graph[root].items():
        push(queue, dist[idx], child, np.int32(1))
    while queue:
        distance, point, depth = pop(queue)
        if point in children:
            continue
        children[point] = distance
        if depth == max_depth:
            continue
        for child, dist in core_graph[point].items():
            if child == root or child in children:
                continue
            push(queue, distance + dist[idx], child, depth + 1)
    return children


@nb.njit(parallel=True)
def manifold_knn_hop(distances, indices, max_depth=2):
    result = [
        {np.int32(0): np.float32(0.0) for _ in range(0)}
        for _ in range(indices.shape[0])
    ]
    for root in nb.prange(indices.shape[0]):
        result[root] = manifold_knn_path(root, distances, indices, max_depth=max_depth)

    return to_csr(result)


@nb.njit(parallel=True)
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
    if use_reachability:
        for point in nb.prange(num_points):
            for child, value in core_graph[point].items():
                core_graph[point][child] = (
                    np.sqrt(rdist(data[point], data[child])),
                    value[1],
                )
    result = [
        {np.int32(0): np.float32(0.0) for _ in range(0)}
        for _ in range(indices.shape[0])
    ]
    for root in nb.prange(indices.shape[0]):
        result[root] = manifold_mst_path(
            root, core_graph, max_depth=max_depth, use_reachability=use_reachability
        )
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
def metric_knn_path(
    root, data, indices, core_distances, max_depth=2, use_reachability=True
):
    """Depth first depth-limit traversal."""
    stack = make_queue()
    enqueued = {root}
    for child in indices[root]:
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
        for child in indices[point]:
            if child in enqueued:
                continue
            distance = np.sqrt(rdist(data[root], data[child]))
            if use_reachability:
                distance = max(distance, core_distances[root], core_distances[child])
            push(stack, distance, child, depth + 1)
            enqueued.add(child)
    return children


@nb.njit()
def metric_mst_path(
    root, data, core_graph, core_distances, max_depth=2, use_reachability=True
):
    """Depth first depth-limit traversal."""
    stack = make_queue()
    enqueued = {root}
    for child in core_graph[root].keys():
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
        for child in core_graph[point].keys():
            if child in enqueued:
                continue
            distance = np.sqrt(rdist(data[root], data[child]))
            if use_reachability:
                distance = max(distance, core_distances[root], core_distances[child])
            push(stack, distance, child, depth + 1)
            enqueued.add(child)
    return children


@nb.njit(parallel=True)
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
    for root in nb.prange(indices.shape[0]):
        result[root] = metric_knn_path(
            root,
            data,
            indices,
            core_distances,
            max_depth=max_depth,
            use_reachability=use_reachability,
        )
    return to_csr(result)


@nb.njit(parallel=True)
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
    for root in nb.prange(indices.shape[0]):
        result[root] = metric_mst_path(
            root,
            data,
            core_graph,
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
