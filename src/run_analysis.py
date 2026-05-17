
"""Command-line analysis workflow.

Example:
    python -m plant_clustering.run_analysis --sample-size 2000 --n-clusters 11
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import cluster_species_table, find_species_clusters, region_summary_for_clusters
from .clustering import assign_clusters, scipy_linkage, summarize_cluster_sizes
from .data import build_binary_matrix, load_plants_data, sample_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster plant species by regional occurrence.")
    parser.add_argument("--data-path", type=Path, default=Path("data/plants.data"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--sample-size", type=int, default=2000,
                        help="Number of records to cluster. Use 0 for all records.")
    parser.add_argument("--n-clusters", type=int, default=11)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plant_names, region_lists = load_plants_data(args.data_path)

    if args.sample_size:
        plant_names, region_lists = sample_records(
            plant_names,
            region_lists,
            args.sample_size,
            random_state=args.random_state,
        )

    binary_matrix = build_binary_matrix(plant_names, region_lists)
    Z = scipy_linkage(binary_matrix)
    labels = assign_clusters(Z, args.n_clusters)

    cluster_sizes = summarize_cluster_sizes(labels)
    cluster_sizes.to_csv(args.output_dir / "cluster_sizes.csv", header=["species_count"])

    top_cluster_ids = cluster_sizes[cluster_sizes > 100].index.tolist()
    if not top_cluster_ids:
        top_cluster_ids = cluster_sizes.sort_values(ascending=False).head(5).index.tolist()

    top_species = cluster_species_table(plant_names, labels, top_cluster_ids)
    top_species.to_csv(args.output_dir / "top_cluster_species_sample.csv", index=False)

    summary = region_summary_for_clusters(binary_matrix, labels, top_cluster_ids)
    summary.to_csv(args.output_dir / "cluster_region_summary.csv", index=False)

    species_to_find = [
        "allium yosemitense",
        "sabal palmetto",
        "cycas revoluta",
        "rosa gallica",
        "lagerstroemia",
        "syringa",
        "hibiscus syriacus",
        "huperzia lucidula",
        "huperzia nutans",
    ]
    species_clusters = find_species_clusters(plant_names, labels, species_to_find)
    species_clusters.to_csv(args.output_dir / "selected_species_clusters.csv", index=False)

    print("Analysis complete.")
    print(f"Plants analyzed: {len(plant_names)}")
    print(f"Regions represented: {binary_matrix.shape[1]}")
    print("Cluster sizes:")
    print(cluster_sizes)


if __name__ == "__main__":
    main()
