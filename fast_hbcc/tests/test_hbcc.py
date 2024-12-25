"""
Tests for HBCC clustering algorithm
Shamelessly based on (i.e. ripped off from) the fast_hdbscan test code
"""

import pytest
import numpy as np
from sklearn.utils.estimator_checks import check_estimator

from sklearn.utils import shuffle
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler

from fast_hbcc import HBCC, fast_hbcc


n_clusters = 3
X, y = make_blobs(n_samples=200, random_state=10)
X, y = shuffle(X, y, random_state=7)
X = StandardScaler().fit_transform(X)

X_missing_data = X.copy()
X_missing_data[0] = [np.nan, 1]
X_missing_data[5] = [np.nan, np.nan]


# --- Tests


def test_missing_data():
    """Tests if nan data are treated as infinite distance from all other points
    and assigned to -1 cluster"""
    model = HBCC().fit(X_missing_data)
    assert model.labels_[0] == -1
    assert model.labels_[5] == -1
    assert model.probabilities_[0] == 0
    assert model.probabilities_[5] == 0
    assert model.probabilities_[5] == 0
    clean_indices = list(range(1, 5)) + list(range(6, 200))
    clean_model = HBCC().fit(X_missing_data[clean_indices])
    assert np.allclose(clean_model.labels_, model.labels_[clean_indices])


def test_hbcc_feature_vector():
    labels, p, ltree, ctree, mtree, graph, bc, neighbors, cdists = fast_hbcc(
        X, return_trees=True
    )
    n_clusters_1 = len(set(labels)) - int(-1 in labels)
    assert n_clusters_1 == n_clusters

    labels = HBCC().fit(X).labels_
    n_clusters_2 = len(set(labels)) - int(-1 in labels)
    assert n_clusters_2 == n_clusters


def test_hbcc_dbscan_clustering():
    clusterer = HBCC().fit(X)
    labels = clusterer.dbscan_clustering(1 / 1.11)
    n_clusters_1 = len(set(labels)) - int(-1 in labels)
    assert n_clusters == n_clusters_1


def test_hbcc_no_clusters():
    labels, p = fast_hbcc(X, min_cluster_size=len(X) + 1)
    n_clusters_1 = len(set(labels)) - int(-1 in labels)
    assert n_clusters_1 == 0

    labels = HBCC(min_cluster_size=len(X) + 1).fit(X).labels_
    n_clusters_2 = len(set(labels)) - int(-1 in labels)
    assert n_clusters_2 == 0


def test_hbcc_min_cluster_size():
    for min_cluster_size in range(2, len(X) + 1, 1):
        labels, p = fast_hbcc(X, min_cluster_size=min_cluster_size)
        true_labels = [label for label in labels if label != -1]
        if len(true_labels) != 0:
            assert np.min(np.bincount(true_labels)) >= min_cluster_size

        labels = HBCC(min_cluster_size=min_cluster_size).fit(X).labels_
        true_labels = [label for label in labels if label != -1]
        if len(true_labels) != 0:
            assert np.min(np.bincount(true_labels)) >= min_cluster_size


def test_hbcc_input_lists():
    X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    HBCC(min_samples=2).fit(X)  # must not raise exception


def test_hbcc_badargs():
    with pytest.raises(ValueError):
        fast_hbcc("fail")
    with pytest.raises(ValueError):
        fast_hbcc(None)
    with pytest.raises(ValueError):
        fast_hbcc(X, min_cluster_size="fail")
    with pytest.raises(ValueError):
        fast_hbcc(X, min_samples="fail")
    with pytest.raises(ValueError):
        fast_hbcc(X, min_samples=-1)
    with pytest.raises(ValueError):
        fast_hbcc(X, min_samples=1)
    with pytest.raises(ValueError):
        fast_hbcc(X, min_samples=X.shape[0])
    with pytest.raises(ValueError):
        fast_hbcc(X, cluster_selection_epsilon="fail")
    with pytest.raises(ValueError):
        fast_hbcc(X, cluster_selection_epsilon=-1)
    with pytest.raises(ValueError):
        fast_hbcc(X, cluster_selection_epsilon=-0.1)
    with pytest.raises(ValueError):
        fast_hbcc(X, cluster_selection_persistence="fail")
    with pytest.raises(ValueError):
        fast_hbcc(X, cluster_selection_persistence=1)
    with pytest.raises(ValueError):
        fast_hbcc(X, cluster_selection_persistence=-0.1)
    with pytest.raises(ValueError):
        fast_hbcc(X, cluster_selection_method="fail")
    with pytest.raises(ValueError):
        fast_hbcc(X, ss_algorithm="fail")


def test_hbcc_allow_single_cluster_with_epsilon():
    np.random.seed(0)
    no_structure = np.random.rand(150, 2)
    c = HBCC(
        min_cluster_size=5,
        cluster_selection_epsilon=0.0,
    ).fit(no_structure)
    unique_labels, counts = np.unique(c.labels_, return_counts=True)
    assert len(unique_labels) == 11
    assert counts[unique_labels == -1] == 30

    c = HBCC(min_cluster_size=5, cluster_selection_epsilon=1 / 1.05).fit(no_structure)
    unique_labels, counts = np.unique(c.labels_, return_counts=True)
    assert len(unique_labels) == 2
    assert counts[unique_labels == -1] == 4


def test_hbcc_max_cluster_size():
    model = HBCC(max_cluster_size=15).fit(X)
    assert len(set(model.labels_)) == 1
    for label in set(model.labels_):
        if label != -1:
            assert np.sum(model.labels_ == label) <= 15


def test_hbcc_persistence_threshold():
    model = HBCC(
        min_cluster_size=5,
        cluster_selection_method="leaf",
        cluster_selection_persistence=20.0,
    ).fit(X)
    assert np.all(model.labels_ == -1)


# Disable for now -- need to refactor to meet newer standards
@pytest.mark.skip(reason="need to refactor to meet newer standards")
def test_hbcc_is_sklearn_estimator():
    check_estimator(HBCC())
