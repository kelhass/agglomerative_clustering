
"""Hierarchical clustering utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage


def jaccard_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """Compute Jaccard distance between two binary vectors."""
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    intersection = np.logical_and(x1 == 1, x2 == 1).sum()
    union = np.logical_or(x1 == 1, x2 == 1).sum()
    if union == 0:
        return 0.0
    return 1.0 - intersection / union


def distance_matrix(X: np.ndarray) -> np.ndarray:
    """Compute a full pairwise Jaccard distance matrix."""
    X = np.asarray(X)
    n = X.shape[0]
    distances = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            d = jaccard_distance(X[i], X[j])
            distances[i, j] = distances[j, i] = d

    return distances


def agglomerative_average_linkage(X: np.ndarray, *, verbose: bool = False) -> np.ndarray:
    """Scratch implementation of agglomerative clustering with average linkage.

    This implementation is useful for learning and validation on small samples.
    For full-size datasets, use scipy_linkage() for computational efficiency.
    """
    X = np.asarray(X)
    n = X.shape[0]
    if n < 2:
        raise ValueError("At least two observations are required.")

    dm = distance_matrix(X)
    clusters = set(range(n))
    cluster_size = {i: 1 for i in range(n)}
    next_id = n
    merges = []

    dist = {}
    for i in range(n):
        for j in range(i + 1, n):
            dist[(i, j)] = dm[i, j]

    while len(clusters) > 1:
        merge_pair = None
        merge_dist = float("inf")

        for (i, j), d in dist.items():
            if i in clusters and j in clusters and d < merge_dist:
                merge_dist = d
                merge_pair = (i, j)

        if merge_pair is None:
            raise RuntimeError("Unable to identify next cluster merge.")

        a, b = merge_pair
        new_id = next_id
        next_id += 1

        new_size = cluster_size[a] + cluster_size[b]
        merges.append([a, b, merge_dist, new_size])
        clusters.remove(a)
        clusters.remove(b)

        for c in list(clusters):
            d_ac = dist[(min(a, c), max(a, c))]
            d_bc = dist[(min(b, c), max(b, c))]
            d_newc = (cluster_size[a] * d_ac + cluster_size[b] * d_bc) / new_size
            dist[(min(new_id, c), max(new_id, c))] = d_newc

        clusters.add(new_id)
        cluster_size[new_id] = new_size

        if verbose and len(merges) % 100 == 0:
            print(
                f"[merge {len(merges)}] merge ({a}, {b}) "
                f"at distance={merge_dist:.4f}; active clusters={len(clusters)}"
            )

    return np.array(merges, dtype=float)


def scipy_linkage(binary_matrix: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Run average-linkage hierarchical clustering with Jaccard distance."""
    return linkage(np.asarray(binary_matrix), method="average", metric="jaccard")


def assign_clusters(linkage_matrix: np.ndarray, n_clusters: int) -> np.ndarray:
    """Convert a linkage matrix into flat cluster labels."""
    return fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")


def summarize_cluster_sizes(labels: np.ndarray) -> pd.Series:
    """Return cluster sizes indexed by cluster id."""
    return pd.Series(labels).value_counts().sort_index()
