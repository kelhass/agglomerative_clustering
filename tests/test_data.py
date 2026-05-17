from pathlib import Path

from plant_clustering.data import build_binary_matrix, load_plants_data, sample_records


def test_load_plants_data(tmp_path):
    file_path = tmp_path / "plants.data"
    file_path.write_text("plant a,ca,or\nplant b,ny\n", encoding="utf-8")

    names, regions = load_plants_data(file_path)

    assert names == ["plant a", "plant b"]
    assert regions == [["ca", "or"], ["ny"]]


def test_build_binary_matrix():
    names = ["plant a", "plant b"]
    regions = [["ca", "or"], ["ny"]]

    matrix = build_binary_matrix(names, regions)

    assert matrix.loc["plant a", "ca"] == 1
    assert matrix.loc["plant a", "ny"] == 0
    assert matrix.loc["plant b", "ny"] == 1


def test_sample_records_reproducible():
    names = ["a", "b", "c", "d"]
    regions = [["ca"], ["or"], ["ny"], ["tx"]]

    sample_1 = sample_records(names, regions, 2, random_state=42)
    sample_2 = sample_records(names, regions, 2, random_state=42)

    assert sample_1 == sample_2
    assert len(sample_1[0]) == 2
