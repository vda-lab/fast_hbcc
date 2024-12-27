import numpy as np
from typing import Literal
from fast_hdbscan.sub_clusters import SubClusterDetector, find_sub_clusters

from .hbcc import compute_boundary_coefficient


def make_boundary_callback(
    num_hops: int = 2,
    hop_type: Literal["manifold", "metric"] = "manifold",
    boundary_connectivity: Literal["knn", "core"] = "knn",
    boundary_use_reachability: bool = True,
):
    return lambda data, _, neighbors, core_distances, min_spanning_tree, points: compute_boundary_coefficient(
        data[points, :],
        neighbors,
        core_distances,
        min_spanning_tree,
        num_hops=num_hops,
        hop_type=hop_type,
        boundary_connectivity=boundary_connectivity,
        boundary_use_reachability=boundary_use_reachability,
    )


def find_boundary_subclusters(
    clusterer,
    cluster_labels=None,
    cluster_probabilities=None,
    num_hops: int = 2,
    hop_type: Literal["manifold", "metric"] = "manifold",
    boundary_connectivity: Literal["knn", "core"] = "knn",
    boundary_use_reachability: bool = True,
    min_cluster_size: int | None = None,
    max_cluster_size: int | None = None,
    allow_single_cluster: bool | None = None,
    cluster_selection_method: Literal["eom", "leaf"] | None = None,
    cluster_selection_epsilon: float = 0.0,
    cluster_selection_persistence: float = 0.0,
):
    return find_sub_clusters(
        clusterer,
        cluster_labels,
        cluster_probabilities,
        lens_callback=make_boundary_callback(num_hops, hop_type, boundary_connectivity, boundary_use_reachability),
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
        allow_single_cluster=allow_single_cluster,
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_persistence=cluster_selection_persistence,
    )



class BoundaryClusterDetector(SubClusterDetector):
    """
    Performs a post-processing step to detect boundary clusters within
    HDBSCAN clusters. The process follows [1]_ but uses the boundary
    coefficient as distance, rather than centrality.

    References
    ----------
    .. [1] Bot, D. M., Peeters, J., Liesenborgs J., & Aerts, J. (2023, November).
       FLASC: A Flare-Sensitive Clustering Algorithm: Extending HDBSCAN* for
       Detecting Branches in Clusters. arXiv:2311.15887.
    """
    def __init__(
        self,
        num_hops: int = 2,
        hop_type: Literal["manifold", "metric"] = "manifold",
        boundary_connectivity: Literal["knn", "core"] = "knn",
        boundary_use_reachability: bool = True,
        min_cluster_size: int | None = None,
        max_cluster_size: int | None = None,
        allow_single_cluster: bool | None = None,
        cluster_selection_method: Literal["eom", "leaf"] | None = None,
        cluster_selection_epsilon: float = 0.0,
        cluster_selection_persistence: float = 0.0,
    ):
        super().__init__(
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
            allow_single_cluster=allow_single_cluster,
            cluster_selection_method=cluster_selection_method,
            cluster_selection_epsilon=cluster_selection_epsilon,
            cluster_selection_persistence=cluster_selection_persistence,
        )
        self.num_hops = num_hops
        self.hop_type = hop_type
        self.boundary_connectivity = boundary_connectivity
        self.boundary_use_reachability = boundary_use_reachability


    def fit(self, clusterer, labels=None, probabilities=None):
        super().fit(clusterer, labels, probabilities, make_boundary_callback(
            self.num_hops, self.hop_type, self.boundary_connectivity, self.boundary_use_reachability
        ))
        self.boundary_coefficient_ = self.lens_values_
        return self