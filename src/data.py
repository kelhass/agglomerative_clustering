
"""Data loading and preprocessing utilities for the UCI Plants dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def load_plants_data(path: str | Path) -> tuple[list[str], list[list[str]]]:
    """Load plant names and region lists from the raw plants.data file.

    Each row in the raw file is comma-separated, with the plant name first
    and the states/provinces where that plant occurs in the remaining fields.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    plant_names: list[str] = []
    region_lists: list[list[str]] = []

    with path.open("r", encoding="iso-8859-1", errors="replace") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(",") if part.strip()]
            if not parts:
                continue
            plant_names.append(parts[0])
            region_lists.append(parts[1:])

    return plant_names, region_lists


def build_binary_matrix(
    plant_names: list[str],
    region_lists: list[list[str]],
    *,
    use_sklearn: bool = True,
) -> pd.DataFrame:
    """Convert plant-region records into a binary feature matrix.

    Rows represent plant species and columns represent geographic regions.
    Values are 1 when a plant occurs in a region and 0 otherwise.
    """
    if len(plant_names) != len(region_lists):
        raise ValueError("plant_names and region_lists must have the same length.")

    if use_sklearn:
        mlb = MultiLabelBinarizer()
        matrix = mlb.fit_transform(region_lists)
        return pd.DataFrame(matrix, index=plant_names, columns=mlb.classes_)

    unique_regions = sorted({region for regions in region_lists for region in regions})
    region_to_col = {region: i for i, region in enumerate(unique_regions)}
    matrix = np.zeros((len(plant_names), len(unique_regions)), dtype=int)

    for row, regions in enumerate(region_lists):
        for region in regions:
            matrix[row, region_to_col[region]] = 1

    return pd.DataFrame(matrix, index=plant_names, columns=unique_regions)


def sample_records(
    plant_names: list[str],
    region_lists: list[list[str]],
    sample_size: int,
    *,
    random_state: int = 42,
) -> tuple[list[str], list[list[str]]]:
    """Return a reproducible random sample of plant records."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")

    rng = np.random.default_rng(random_state)
    n = len(plant_names)
    sample_size = min(sample_size, n)
    indices = rng.choice(n, size=sample_size, replace=False)

    return [plant_names[i] for i in indices], [region_lists[i] for i in indices]
