
import numpy as np

from plant_clustering.clustering import (
    agglomerative_average_linkage,
    assign_clusters,
    distance_matrix,
    jaccard_distance,
    scipy_linkage,
)


def test_jaccard_distance():
    x1 = np.array([1, 1, 0, 0])
    x2 = np.array([1, 0, 1, 0])

    assert np.isclose(jaccard_distance(x1, x2), 2 / 3)


def test_distance_matrix_is_symmetric():
    X = np.array([[1, 0], [1, 1], [0, 1]])
    dm = distance_matrix(X)

    assert dm.shape == (3, 3)
    assert np.allclose(dm, dm.T)
    assert np.allclose(np.diag(dm), 0)


def test_scratch_agglomerative_shape():
    X = np.array([[1, 0], [1, 1], [0, 1]])
    Z = agglomerative_average_linkage(X)

    assert Z.shape == (2, 4)


def test_scipy_linkage_and_flat_clusters():
    X = np.array([[1, 0], [1, 1], [0, 1], [0, 1]])
    Z = scipy_linkage(X)
    labels = assign_clusters(Z, 2)

    assert Z.shape == (3, 4)
    assert len(labels) == 4
