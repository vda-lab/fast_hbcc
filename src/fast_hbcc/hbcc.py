import numba as nb
import numpy as np
from typing import Literal
from scipy.sparse import csr_array
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted, _check_sample_weight
from fast_hdbscan.core_graph import (
    core_graph_clusters,
    core_graph_to_edge_list,
    CoreGraph,
)
from fast_hdbscan.hdbscan import (
    HDBSCAN,
    to_numpy_rec_array,
    remap_condensed_tree,
    remap_single_linkage_tree,
    compute_minimum_spanning_tree,
)


from .n_hop import manifold_n_hop, metric_n_hop


def check_greater_equal(min_value=1, datatype=np.integer, **kwargs):
    for k, v in kwargs.items():
        if not (np.issubdtype(type(v), datatype) and v >= min_value):
            raise ValueError(
                f"{k} must be a{'n integer' if datatype is np.integer else ' float' } "
                f"greater or equal to {min_value}, {v} given."
            )


def check_optional_size(size, **kwargs):
    for k, v in kwargs.items():
        if v is not None and v.shape[0] != size:
            raise ValueError(
                f"{k} must have the same number of samples as data, "
                f"{v.shape[0]} for {size} samples given."
            )


def check_literals(**kwargs):
    for k, (v, allowed) in kwargs.items():
        if v not in allowed:
            raise ValueError(f"Invalid {k} {v}")


def remap_csr_graph(graph, finite_index, internal_to_raw, num_points):
    new_indptr = np.empty(num_points + 1, dtype=graph.indptr.dtype)
    new_indptr[: finite_index[0] + 1] = 0
    for idx, (start, end) in enumerate(zip(finite_index, finite_index[1:])):
        new_indptr[start + 1 : end + 1] = graph.indptr[idx + 1]
        start = end
    new_indptr[finite_index[-1] + 1 :] = graph.indptr[-1]
    new_indices = np.vectorize(internal_to_raw.get)(graph.indices)
    return new_indices, new_indptr


def remap_core_graph(graph, finite_index, internal_to_raw, num_points):
    new_indices, new_indptr = remap_csr_graph(
        graph, finite_index, internal_to_raw, num_points
    )
    return CoreGraph(graph.weights, graph.distances, new_indices, new_indptr)


def boundary_coefficient_from_csr(g):
    g.sort_indices()
    g_inv = g.copy()
    g_inv.data = 1 / g_inv.data
    degree = np.diff(g.indptr)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (
            g.sum(axis=1) * g_inv.sum(axis=1)
            - ((g_inv @ g.power(2)) * g_inv).sum(axis=1) / 2
        ) / (degree * degree)
    return np.where(np.isnan(result), np.inf, result)


def compute_boundary_coefficient(
    data,
    neighbors,
    core_distances,
    min_spanning_tree,
    num_hops: int = 2,
    hop_type: Literal["manifold", "metric"] = "manifold",
    boundary_connectivity: Literal["knn", "core"] = "knn",
    boundary_use_reachability: bool = True,
):
    n_hop_expansion = manifold_n_hop if hop_type == "manifold" else metric_n_hop
    manifold = csr_array(
        n_hop_expansion(
            data,
            neighbors,
            core_distances,
            min_spanning_tree,
            max_depth=num_hops,
            connectivity=boundary_connectivity,
            use_reachability=boundary_use_reachability,
        ),
        shape=(data.shape[0], data.shape[0]),
    )

    return boundary_coefficient_from_csr(manifold)


def fast_hbcc(
    data,
    data_labels=None,
    sample_weights=None,
    num_hops: int = 2,
    min_samples: int = 5,
    min_cluster_size: int = 25,
    hop_type: Literal["manifold", "metric"] = "manifold",
    boundary_connectivity: Literal["knn", "core"] = "knn",
    boundary_use_reachability: bool = True,
    cluster_selection_method: Literal["eom", "leaf"] = "eom",
    allow_single_cluster: bool = False,
    max_cluster_size: float = np.inf,
    cluster_selection_epsilon: float = 0.0,
    cluster_selection_persistence: float = 0.0,
    ss_algorithm: Literal["bc", "bc_simple"] = "bc",
    return_trees: bool = False,
    num_jobs: int = 0,
):
    data = check_array(data)

    # Validate parameter values
    check_greater_equal(
        min_value=2,
        datatype=np.integer,
        min_samples=min_samples,
        min_cluster_size=min_cluster_size,
    )
    if min_samples >= data.shape[0]:
        raise ValueError(
            f"min_samples must be less than the number of samples: {data.shape[0]}."
        )

    check_greater_equal(
        min_value=0,
        datatype=np.integer,
        num_hops=num_hops,
    )
    max_cluster_size = float(max_cluster_size)
    check_greater_equal(
        min_value=0.0,
        datatype=np.floating,
        max_cluster_size=max_cluster_size,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_persistence=cluster_selection_persistence,
    )
    check_optional_size(
        data.shape[0], data_labels=data_labels, sample_weights=sample_weights
    )
    check_literals(
        hop_type=(hop_type, ["manifold", "metric"]),
        boundary_connectivity=(boundary_connectivity, ["knn", "core"]),
        cluster_selection_method=(cluster_selection_method, ["eom", "leaf"]),
        ss_algorithm=(ss_algorithm, ["bc", "bc_simple"]),
    )

    original_num_threads = nb.get_num_threads()
    if num_jobs <= 0:
        num_jobs = original_num_threads + num_jobs
    nb.set_num_threads(num_jobs)

    minimum_spanning_tree, neighbors, core_distances = compute_minimum_spanning_tree(
        data, min_samples=min_samples, sample_weights=sample_weights
    )

    boundary_coefficient = compute_boundary_coefficient(
        data,
        neighbors,
        core_distances,
        minimum_spanning_tree,
        num_hops=num_hops,
        hop_type=hop_type,
        boundary_connectivity=boundary_connectivity,
        boundary_use_reachability=boundary_use_reachability,
    )

    result = core_graph_clusters(
        boundary_coefficient,
        neighbors,
        core_distances,
        minimum_spanning_tree,
        data_labels=data_labels,
        sample_weights=sample_weights,
        min_cluster_size=min_cluster_size,
        cluster_selection_method=cluster_selection_method,
        allow_single_cluster=allow_single_cluster,
        max_cluster_size=max_cluster_size,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_persistence=cluster_selection_persistence,
        semi_supervised=data_labels is not None,
        ss_algorithm=ss_algorithm,
    )

    nb.set_num_threads(original_num_threads)
    return (
        *result,
        boundary_coefficient,
        neighbors,
        core_distances,
    )[: (None if return_trees else 2)]


class HBCC(HDBSCAN):
    """
    An SKLEARN-style estimator for computing Hierarchical Boundary Coefficient
    Clustering (HBCC). This algorithm is inspired by the work of Peng et al.
    [4]_, uses algorithms from Vandaele et al. [2]_, Campello et al. [1]_,
    McInnes et al. [3]_ and relies on code from
    `fast_hdbscan <https://github.com/TutteInstitute/fast_hdbscan>`__.

    The algorithm contains the following steps:

    1. Compute k nearest neighbors and minimum spanning tree.
    2. Compute the boundary coefficient for each point.
    3. Compute minimum spanning tree from boundary coefficient weighted knn--mst
       graph union.
    5. Compute HDBSCAN cluster hierarchy and selection.


    Parameters
    ----------
    num_hops: int, default=2
        The number of hops used to expand the boundary coefficient graph.
    min_samples: int, default=5
        Core distance is computed as the min_samples-nearest neighbor distance.
    min_cluster_size: int, default=25
        The minimum number of samples in a cluster.
    hop_type: 'manifold' or 'metric', default='manifold'
        The type of hop expansion to use. Manifold adds edge distances on
        traversal, metric computes distance between visited points.
    boundary_connectivity: 'knn' or 'core', default='knn'
        Which graph to compute the boundary coefficient on. 'knn' uses the
        k-nearest neighbors graph, 'core' uses the knn--mst union graph.
    boundary_use_reachability: boolean, default=False
        Whether to use mutual reachability or raw distances for the boundary
        coefficient computation.
    cluster_selection_method: 'eom' or 'leaf', default='eom'
        HDBSCAN cluster selection strategy.
    allow_single_cluster: bool, default=False
        HDBSCAN cluster selection parameter controlling whether to allow single
        cluster during selection.
    max_cluster_size: int, default=np.inf
        HDBSCAN cluster selection parameter limiting the maximum cluster size.
    cluster_selection_epsilon: float, default=0.0
        HDBSCAN cluster selection parameter controlling minimum cluster death
        distance.
    cluster_selection_persistence: float, default=0.0
        HDBSCAN cluster selection parameter controlling minimum cluster
        persistence.
    ss_algorithm: 'bc' or 'bc_simple', default='bc'
        HDBSCAN clustering selection parameter controlling the semi-supervised
        strategy.
    num_jobs : int, optional (default=0)
        The number of threads to use for the computation. Zero means using all
        threads. Negative values indicate all but that number of threads.

    Attributes
    ----------
    labels_ : numpy.ndarray, shape (n_samples,)
        The computed cluster labels.
    probabilities_: numpy.ndarray, shape (n_samples,)
        The computed cluster probabilities.
    boundary_coefficient_ : numpy.ndarray, shape (n_samples,)
        The computed boundary coefficient for each point.
    condensed_tree_ : hdbscan.plots.CondensedTree
        The condensed cluster hierarchy used to generate clusters.
    single_linkage_tree : hdbscan.plots.SingleLinkageTree
        The single linkage tree produced during clustering in scipy
        hierarchical clustering format
        (see http://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html).
    min_spanning_tree_ : hdbscan.plots.MinimumSpanningTree
        The minimum spanning as an edgelist. If gen_min_span_tree was False
        this will be None.


    References
    ----------
    .. [1] Campello, R. J., Moulavi, D., & Sander, J. (2013, April).
       Density-based clustering based on hierarchical density estimates. In
       Pacific-Asia Conference on Knowledge Discovery and Data Mining (pp.
       160-172). Springer Berlin Heidelberg.

    .. [2] Vandaele, R., Saeys, Y., & De Bie, T. (2019). The Boundary
        Coefficient : a Vertex Measure for Visualizing and Finding Structure in
        Weighted Graphs. 15th International Workshop on Mining and Learning with
        Graphs (MLG).

    .. [3] McInnes, L., & Healy, J. (2017). Accelerated Hierarchical Density
        Based Clustering. 2017 IEEE International Conference on Data Mining
        Workshops (ICDMW), 2017-Novem, 33–42.
        https://doi.org/10.1109/ICDMW.2017.12.

    .. [4] Peng, D., Gui, Z., Wang, D., Ma, Y., Huang, Z., Zhou, Y., & Wu, H.
        (2022). Clustering by measuring local direction centrality for data with
        heterogeneous density and weak connectivity. Nature Communications,
        13(1), 1–14. https://doi.org/10.1038/s41467-022-33136-9.
    """

    def __init__(
        self,
        num_hops: int = 2,
        min_samples: int = 5,
        min_cluster_size: int = 25,
        hop_type: Literal["manifold", "metric"] = "manifold",
        boundary_connectivity: Literal["knn", "core"] = "knn",
        boundary_use_reachability: bool = True,
        cluster_selection_method: Literal["eom", "leaf"] = "eom",
        allow_single_cluster: bool = False,
        max_cluster_size: float = np.inf,
        cluster_selection_epsilon: float = 0.0,
        cluster_selection_persistence: float = 0.0,
        ss_algorithm: Literal["bc", "bc_simple"] = "bc",
        num_jobs: int = 0,
    ):
        super().__init__(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method=cluster_selection_method,
            allow_single_cluster=allow_single_cluster,
            max_cluster_size=max_cluster_size,
            cluster_selection_epsilon=cluster_selection_epsilon,
            cluster_selection_persistence=cluster_selection_persistence,
            ss_algorithm=ss_algorithm,
        )
        self.num_hops = num_hops
        self.hop_type = hop_type
        self.boundary_connectivity = boundary_connectivity
        self.boundary_use_reachability = boundary_use_reachability
        self.num_jobs = num_jobs

    def fit(self, X, y=None, sample_weight=None, **fit_params):
        """
        Computes the Hierarchical Boundary Coefficient Clustering (HBCC).

        Parameters
        ----------
        X: float[:, ::1]
            The data to cluster.
        y: int[::1], optional
            Datapoint labels for semi-supervised clustering.
        **fit_params: dict
            Ignored.

        Returns
        -------
        self: HBCC
            The fitted estimator.
        """
        if y is not None:
            self.semi_supervised = True
            X, y = check_X_y(X, y, accept_sparse="csr", force_all_finite=False)
            self._raw_labels = y
            # Replace non-finite labels with -1 labels
            y[~np.isfinite(y)] = -1

            if ~np.any(y != -1):
                raise ValueError(
                    "y must contain at least one label > -1. Currently it only contains -1 and/or non-finite labels!"
                )
        else:
            X = check_array(X, accept_sparse="csr", force_all_finite=False)
            self._raw_data = X
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=np.float32)

        self._all_finite = np.all(np.isfinite(X))
        if ~self._all_finite:
            # Pass only the purely finite indices into hdbscan
            # We will later assign all non-finite points to the background -1 cluster
            finite_index = np.where(np.isfinite(X).sum(axis=1) == X.shape[1])[0]
            clean_data = X[finite_index]
            clean_labels = y[finite_index] if self.semi_supervised else None
            sample_weight = (
                sample_weight[finite_index] if sample_weight is not None else None
            )

            internal_to_raw = {
                x: y for x, y in zip(range(len(finite_index)), finite_index)
            }
            outliers = list(set(range(X.shape[0])) - set(finite_index))
        else:
            clean_data = X
            clean_labels = y

        kwargs = self.get_params()
        (
            self.labels_,
            self.probabilities_,
            self._single_linkage_tree,
            self._condensed_tree,
            self._min_spanning_tree,
            self._core_graph,
            self.boundary_coefficient_,
            self._neighbors,
            self._core_distances,
        ) = fast_hbcc(
            clean_data,
            clean_labels,
            sample_weight,
            return_trees=True,
            **kwargs,
        )
        self._condensed_tree = to_numpy_rec_array(self._condensed_tree)

        if not self._all_finite:
            # remap indices to align with original data in the case of non-finite entries.
            self._condensed_tree = remap_condensed_tree(
                self._condensed_tree, internal_to_raw, outliers
            )
            self._single_linkage_tree = remap_single_linkage_tree(
                self._single_linkage_tree, internal_to_raw, outliers
            )
            self._core_graph = remap_core_graph(
                self._core_graph, finite_index, internal_to_raw, X.shape[0]
            )

            new_labels = np.full(X.shape[0], -1)
            new_labels[finite_index] = self.labels_
            self.labels_ = new_labels

            new_probabilities = np.zeros(X.shape[0])
            new_probabilities[finite_index] = self.probabilities_
            self.probabilities_ = new_probabilities

            new_bc = np.zeros(X.shape[0])
            new_bc[finite_index] = self.boundary_coefficient_
            self.boundary_coefficient_ = new_bc

        return self

    @property
    def approximation_graph_(self):
        """See :class:`~hdbscan.plots.ApproximationGraph` for documentation."""
        from hdbscan.plots import ApproximationGraph

        check_is_fitted(
            self,
            "_core_graph",
            msg="You first need to fit the BranchDetector model before accessing the approximation graphs",
        )

        return ApproximationGraph(
            [core_graph_to_edge_list(self._core_graph)],
            self.labels_,
            self.probabilities_,
            self.boundary_coefficient_,
            lens_name="boundary_coefficient",
            raw_data=self._raw_data,
        )
