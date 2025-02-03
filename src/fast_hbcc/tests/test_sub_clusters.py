import pytest
import numpy as np
from sklearn.exceptions import NotFittedError
from fast_hdbscan import HDBSCAN

from fast_hbcc import BoundaryClusterDetector, find_boundary_sub_clusters

try:
    from hdbscan.plots import ApproximationGraph, CondensedTree, SingleLinkageTree

    HAVE_HDBSCAN = True
except ModuleNotFoundError:
    HAVE_HDBSCAN = False


def make_branches(points_per_branch=30):
    # Control points for line segments that merge three clusters
    p0 = (0.13, -0.26)
    p1 = (0.24, -0.12)
    p2 = (0.32, 0.1)
    p3 = (0.13, 0.1)

    # Noisy points along lines between three clusters
    return np.concatenate(
        [
            np.column_stack(
                (
                    np.linspace(p_start[0], p_end[0], points_per_branch),
                    np.linspace(p_start[1], p_end[1], points_per_branch),
                )
            )
            + np.random.normal(size=(points_per_branch, 2), scale=0.01)
            for p_start, p_end in [(p0, p1), (p1, p2), (p1, p3)]
        ]
    )


np.random.seed(0)
X = np.concatenate(
    (
        make_branches(),
        make_branches()[:60] + np.array([0.3, 0]),
    )
)
c = HDBSCAN(min_samples=5, min_cluster_size=10).fit(X)


def check_detected_groups(c, n_clusters=3, n_subs=6, overridden=False):
    """Checks BoundaryClusterDetector output for main invariants."""
    noise_mask = c.labels_ == -1
    assert np.all(np.unique(c.labels_[~noise_mask]) == np.arange(n_subs))
    assert np.all(np.unique(c.cluster_labels_[~noise_mask]) == np.arange(n_clusters))
    assert (c.sub_cluster_labels_[noise_mask] == 0).all()
    assert (c.sub_cluster_probabilities_[noise_mask] == 1.0).all()
    assert (c.probabilities_[noise_mask] == 0.0).all()
    assert (c.cluster_probabilities_[noise_mask] == 0.0).all()
    if not overridden:
        assert len(c.cluster_points_) == n_clusters
        for condensed_tree, linkage_tree in zip(c._condensed_trees, c._linkage_trees):
            assert linkage_tree is not None
            assert condensed_tree is not None


# --- Detecting Sub_Clusters


@pytest.mark.skipif(not HAVE_HDBSCAN, reason='requires HDBSCAN')
def test_attributes():
    b = BoundaryClusterDetector().fit(c)
    check_detected_groups(b, n_clusters=2, n_subs=6)
    assert len(b.linkage_trees_) == 2
    assert len(b.condensed_trees_) == 2
    assert isinstance(b.condensed_trees_[0], CondensedTree)
    assert isinstance(b.linkage_trees_[0], SingleLinkageTree)
    assert isinstance(b.approximation_graph_, ApproximationGraph)


def test_selection_method():
    b = BoundaryClusterDetector(cluster_selection_method="eom").fit(c)
    check_detected_groups(b, n_clusters=2, n_subs=6)

    b = BoundaryClusterDetector(cluster_selection_method="leaf").fit(c)
    check_detected_groups(b, n_clusters=2, n_subs=7)


def test_sample_weight():
    np.random.seed(0)
    sample_weight = np.random.uniform(low=0, high=1, size=X.shape[0]).astype(np.float32)
    b = BoundaryClusterDetector().fit(c, sample_weight=sample_weight)
    check_detected_groups(b, n_clusters=2, n_subs=2)


def test_min_cluster_size():
    b = BoundaryClusterDetector(min_cluster_size=7).fit(c)
    labels, counts = np.unique(
        b.labels_[b.sub_cluster_labels_ >= 0], return_counts=True
    )
    assert (counts[labels >= 0] >= 7).all()
    check_detected_groups(b, n_clusters=2, n_subs=7)


def test_max_cluster_size():
    b = BoundaryClusterDetector(max_cluster_size=5).fit(c)
    assert len(set(b.labels_)) == 2


def test_override_cluster_labels():
    X_missing = X.copy()
    X_missing[60:80] = np.nan
    c = HDBSCAN(min_cluster_size=5).fit(X_missing)
    split_y = c.labels_.copy()
    split_y[split_y == 1] = 0
    split_y[split_y == 2] = 1
    b = BoundaryClusterDetector().fit(c, split_y)
    check_detected_groups(b, n_clusters=2, n_subs=4, overridden=True)
    assert b._condensed_trees[0] is None
    assert b._linkage_trees[0] is None


def test_allow_single_cluster_with_filters():
    # Without persistence, find 6 branches
    b = BoundaryClusterDetector(
        min_cluster_size=5,
        cluster_selection_method="leaf",
    ).fit(c)
    unique_labels = np.unique(b.labels_)
    assert len(unique_labels) == 14

    # Adding persistence removes the branches
    b = BoundaryClusterDetector(
        min_cluster_size=5,
        cluster_selection_method="leaf",
        cluster_selection_persistence=0.15,
    ).fit(c)
    unique_labels = np.unique(b.labels_)
    assert len(unique_labels) == 8

    # Adding epsilon removes some branches
    b = BoundaryClusterDetector(
        min_cluster_size=5,
        cluster_selection_method="leaf",
        cluster_selection_epsilon=1 / 0.002,
    ).fit(c)
    unique_labels = np.unique(b.labels_)
    assert len(unique_labels) == 2


def test_badargs():
    c_nofit = HDBSCAN(min_cluster_size=5)

    with pytest.raises(TypeError):
        find_boundary_sub_clusters("fail")
    with pytest.raises(TypeError):
        find_boundary_sub_clusters(None)
    with pytest.raises(NotFittedError):
        find_boundary_sub_clusters(c_nofit)
    with pytest.raises(ValueError):
        find_boundary_sub_clusters(c, min_cluster_size=-1)
    with pytest.raises(ValueError):
        find_boundary_sub_clusters(c, min_cluster_size=0)
    with pytest.raises(ValueError):
        find_boundary_sub_clusters(c, min_cluster_size=1)
    with pytest.raises(ValueError):
        find_boundary_sub_clusters(c, min_cluster_size=2.0)
    with pytest.raises(ValueError):
        find_boundary_sub_clusters(c, min_cluster_size="fail")
    with pytest.raises(ValueError):
        find_boundary_sub_clusters(c, cluster_selection_persistence=-0.1)
    with pytest.raises(ValueError):
        find_boundary_sub_clusters(c, cluster_selection_epsilon=-0.1)
    with pytest.raises(ValueError):
        find_boundary_sub_clusters(
            c,
            cluster_selection_method="something_else",
        )
