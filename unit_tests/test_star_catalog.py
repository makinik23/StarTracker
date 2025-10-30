import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from star_tracker.star_catalog.star_catalog import StarCatalog, Star


@pytest.fixture
def small_catalog_csv(tmp_path: Path) -> Path:
    data = {
        "id": [1, 2, 3, 4],
        "proper": ["Alpha", "Beta", "Gamma", "Delta"],
        "ra": [0.0, 90.0, 180.0, 270.0],     # [deg]
        "dec": [0.0, 45.0, 0.0, -45.0],      # [deg]
        "mag": [1.0, 5.5, 6.5, 2.0],
    }

    df = pd.DataFrame(data)
    csv_path = tmp_path / "mini_hyg.csv"
    df.to_csv(csv_path, index=False)

    return csv_path

def test_catalog_loads_and_drops_nans(small_catalog_csv):
    """
    Test loading of catalog and dropping of NaN values.
    """
    df = pd.read_csv(small_catalog_csv)
    df.loc[len(df)] = [99, "Bad", np.nan, np.nan, 3.3]
    bad_csv = small_catalog_csv.with_name("mini_hyg_bad.csv")
    df.to_csv(bad_csv, index=False)

    cat = StarCatalog(str(bad_csv))

    assert len(cat.df) == 4
    assert all(col in cat.df.columns for col in ["ra", "dec", "mag"])



def test_vectors_computation_correctness(small_catalog_csv):
    cat = StarCatalog(str(small_catalog_csv))

    vecs = cat.vectors

    assert vecs.shape == (4, 3)

    norms = np.linalg.norm(vecs, axis=1)

    np.testing.assert_allclose(norms, 1.0, atol=1e-10)
    np.testing.assert_allclose(vecs[0], [1.0, 0.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(vecs[1], [0.0, np.sqrt(0.5), np.sqrt(0.5)], atol=1e-10)


def test_vectors_cached_property(small_catalog_csv):
    """
    Test that vectors property is cached and does not recompute on DataFrame change.
    """
    cat = StarCatalog(str(small_catalog_csv))

    first = cat.vectors

    cat.df.loc[0, "ra"] = 999.0
    second = cat.vectors

    assert np.shares_memory(first, second)

    np.testing.assert_allclose(first[0], [1.0, 0.0, 0.0], atol=1e-10)


def test_filtered_returns_only_bright_stars(small_catalog_csv):
    """
    Test filtering by magnitude.
    """
    cat = StarCatalog(str(small_catalog_csv))

    vecs, mags = cat.filtered(max_mag=6.0)
    
    assert vecs.shape[0] == 3
    assert all(m <= 6.0 for m in mags)


def test_cone_search_finds_expected_star(small_catalog_csv):
    """
    Test cone search functionality.
    """
    cat = StarCatalog(str(small_catalog_csv))
    dir_vec = np.array([1.0, 0.0, 0.0])

    res = cat.cone_search(dir_vec, half_angle_deg=10.0)

    assert not res.empty
    assert set(res["id"]) == {1}

    res2 = cat.cone_search(dir_vec, half_angle_deg=120.0)

    assert len(res2) == 3
    assert set(res2["id"]) == {1, 2, 4}

    res3 = cat.cone_search(dir_vec, half_angle_deg=180.0)

    assert len(res3) == 4

    angles = res3["angle_deg"].to_numpy()

    assert np.all(np.diff(angles) >= -1e-12)


def test_cone_search_invariant_to_vector_scaling(small_catalog_csv):
    """
    Test that cone search results are invariant to scaling of the direction vector.
    """
    cat = StarCatalog(str(small_catalog_csv))

    dir_vec = np.array([10.0, 0.0, 0.0])
    res1 = cat.cone_search(dir_vec, half_angle_deg=15.0)
    res2 = cat.cone_search(dir_vec * 3.0, half_angle_deg=15.0)
   
    pd.testing.assert_frame_equal(res1.reset_index(drop=True), res2.reset_index(drop=True))



def test_as_stars_structure_and_types(small_catalog_csv):
    """
    Test that as_stars method returns correct structure and types.
    """
    cat = StarCatalog(str(small_catalog_csv))

    stars = cat.as_stars(max_mag=6.0)

    assert all(isinstance(s, Star) for s in stars)
    assert len(stars) == 3

    s0 = stars[0]

    assert hasattr(s0, "id") and hasattr(s0, "vector")
    assert isinstance(s0.vector, np.ndarray)

    np.testing.assert_allclose(np.linalg.norm(s0.vector), 1.0, atol=1e-10)
