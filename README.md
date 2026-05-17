
# Hierarchical Clustering of Plant Species by Geographic Distribution

This project applies agglomerative hierarchical clustering to the UCI Plants dataset to group plant species based on where they occur geographically. Each plant is represented as a binary vector of U.S. states, Canadian provinces, and other regions. Jaccard distance is used to compare species because the features represent presence or absence across regions.

The project was originally developed as part of GRAD 50400 at Purdue University and has been refactored into a portfolio-ready repository with reusable source code, tests, and a cleaned analysis workflow.

## Project Goals

- Parse the raw plant distribution dataset into a plant-by-region binary matrix.
- Implement core hierarchical clustering concepts, including Jaccard distance and average linkage.
- Compare a scratch implementation with SciPy's hierarchical clustering tools.
- Analyze large clusters to identify geographic patterns in plant distributions.
- Package the project in a reproducible format for portfolio review.

## Repository Structure

```text
.
├── data/
│   ├── README.md
│   └── plants.data
├── notebooks/
│   ├── Project2_Hassett_original.ipynb
│   └── hierarchical_clustering_cleaned.ipynb
├── reports/
│   └── top_cluster_species.csv
├── src/
│   └── plant_clustering/
│       ├── analysis.py
│       ├── clustering.py
│       ├── data.py
│       └── run_analysis.py
├── tests/
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Data

This project uses the public Plants dataset from the UCI Machine Learning Repository. The raw dataset is included in `data/plants.data`.

The original notebook notes that the dataset is publicly available under the Creative Commons Attribution 4.0 International license. See `data/README.md` for attribution and usage notes.

## Methods

The workflow uses the following steps:

1. Load the raw comma-separated plant records.
2. Split each row into a plant name and a list of regions.
3. Convert the data into a binary matrix where each region is a feature.
4. Compute Jaccard distance between binary region vectors.
5. Run agglomerative hierarchical clustering with average linkage.
6. Cut the hierarchy into flat clusters and summarize cluster composition.
7. Interpret clusters by their most common regions and selected species assignments.

## Key Results from the Original Analysis

The original full-dataset analysis clustered 34,781 plant records across 70 geographic regions. The hierarchy was split into 11 flat clusters, and five clusters contained more than 1,000 species.

| Cluster | Number of species | Number of regions | Top region | % in top | 2nd top region | % in 2nd top | 3rd top region | % in 3rd top |
|---:|---:|---:|---|---:|---|---:|---|---:|
| 2 | 3,585 | 30 | pr | 97.2 | vi | 40.5 | fl | 28.2 |
| 3 | 1,991 | 21 | hi | 99.8 | fl | 2.9 | vi | 0.5 |
| 5 | 1,521 | 67 | ak | 61.7 | yt | 41.5 | nt | 37.9 |
| 6 | 17,032 | 70 | ca | 53.8 | az | 29.4 | or | 27.9 |
| 9 | 10,197 | 70 | nc | 56.9 | ga | 56.6 | va | 54.0 |

The results suggest that geography and climate are major drivers of the clusters. Some clusters are dominated by tropical or island regions, while others represent broader temperate or arctic/subarctic distributions.

## How to Run

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

Run the sample analysis workflow:

```bash
PYTHONPATH=src python -m plant_clustering.run_analysis --sample-size 2000 --n-clusters 11
```

The command writes summary outputs to the `reports/` folder. The default sample size is intentionally smaller than the full dataset so the workflow can run on a typical laptop. Use `--sample-size 0` to run on all records if your machine has enough memory.

## Testing

Run the test suite with:

```bash
PYTHONPATH=src pytest
```

## Skills Demonstrated

- Python project organization
- Data parsing and preprocessing
- Binary feature engineering
- Jaccard distance for sparse binary data
- Agglomerative hierarchical clustering
- SciPy and scikit-learn workflows
- Reproducible analysis design
- Writing reusable, tested source code

## Notes

The scratch agglomerative clustering implementation is included for learning and validation on small samples. The full dataset is better handled with optimized scientific Python tools because hierarchical clustering requires substantial pairwise distance computation.

## License

Original code and documentation in this repository are licensed under the MIT License. The included dataset is subject to its own public dataset license; see `data/README.md` for data attribution.
