
"""Analysis helpers for interpreting plant clusters."""

from __future__ import annotations

import pandas as pd


def cluster_species_table(
    plant_names: list[str],
    labels,
    cluster_ids: list[int],
) -> pd.DataFrame:
    """Create a wide table containing species names for selected clusters."""
    cluster_species = {}
    for cluster_id in cluster_ids:
        species = [
            plant_names[i]
            for i, label in enumerate(labels)
            if label == cluster_id
        ]
        cluster_species[f"Cluster {cluster_id}"] = pd.Series(species)

    return pd.DataFrame(cluster_species)


def region_summary_for_clusters(
    binary_matrix: pd.DataFrame,
    labels,
    cluster_ids: list[int],
) -> pd.DataFrame:
    """Summarize top regions and cluster sizes for selected cluster ids."""
    labels_series = pd.Series(labels, index=binary_matrix.index, name="cluster_id")
    rows = []

    for cluster_id in cluster_ids:
        species = labels_series[labels_series == cluster_id].index
        region_counts = binary_matrix.loc[species].sum(axis=0)
        region_counts = region_counts[region_counts > 0].sort_values(ascending=False)

        row = {
            "cluster": cluster_id,
            "number_of_species": len(species),
            "number_of_regions": len(region_counts),
        }

        for rank, (region, count) in enumerate(region_counts.head(3).items(), start=1):
            row[f"top_region_{rank}"] = region
            row[f"top_region_{rank}_count"] = int(count)
            row[f"top_region_{rank}_percent"] = round(100 * count / len(species), 1)

        rows.append(row)

    return pd.DataFrame(rows)


def find_species_clusters(
    plant_names: list[str],
    labels,
    species_queries: list[str],
) -> pd.DataFrame:
    """Find the first matching species and cluster for each query string."""
    cluster_df = pd.DataFrame({"species": plant_names, "cluster_id": labels})
    rows = []

    for query in species_queries:
        matches = cluster_df[
            cluster_df["species"].str.lower().str.contains(query.lower(), na=False)
        ]
        if matches.empty:
            rows.append({"query": query, "matched_species": None, "cluster_id": None})
        else:
            first_match = matches.iloc[0]
            rows.append({
                "query": query,
                "matched_species": first_match["species"],
                "cluster_id": int(first_match["cluster_id"]),
            })

    return pd.DataFrame(rows)
