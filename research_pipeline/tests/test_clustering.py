import numpy as np
import pandas as pd

from research_pipeline.step3_clustering.clustering import (
    _feature_matrix,
    drop_empty_behavior_factors,
    fit_behavior_kmeans,
    select_representatives,
)


def test_kmeans_is_reproducible_and_representatives_are_nearest_to_centers():
    features = pd.DataFrame(
        {"ic_mean": [1.0, 1.1, -1.0, -1.1], "rank_ic_mean": [1.0, 1.1, -1.0, -1.1]},
        index=["f1", "f2", "f3", "f4"],
    )
    first = fit_behavior_kmeans(features, n_clusters=2, random_state=7)
    second = fit_behavior_kmeans(features, n_clusters=2, random_state=7)

    assert np.array_equal(first["labels"], second["labels"])
    assert np.allclose(first["centers"], second["centers"])
    representatives = select_representatives(
        first["labels"], features, first["centers"], per_cluster=1
    )
    assert set(representatives) == {"f1", "f3"}


def test_kmeans_rejects_more_clusters_than_factor_rows():
    features = pd.DataFrame({"x": [1.0, 2.0]}, index=["f1", "f2"])
    try:
        fit_behavior_kmeans(features, n_clusters=3, random_state=1)
    except ValueError as exc:
        assert "n_clusters" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_drop_empty_behavior_factors_keeps_finite_rows():
    features = pd.DataFrame(
        {
            "ic_mean": [1.0, np.nan],
            "rank_ic_mean": [0.5, np.nan],
            "slope_mean": [0.2, np.inf],
        },
        index=["f1", "f2"],
    )
    usable, skipped = drop_empty_behavior_factors(features)

    assert list(usable.index) == ["f1"]
    assert skipped == ["f2"]
    assert usable.loc["f1", "ic_mean"] == 1.0


def test_feature_matrix_still_rejects_empty_rows():
    features = pd.DataFrame(
        {"ic_mean": [1.0, np.nan], "rank_ic_mean": [0.5, np.nan]},
        index=["f1", "f2"],
    )
    try:
        _feature_matrix(features)
    except ValueError as exc:
        assert "each factor needs at least one finite behavior feature" in str(exc)
    else:
        raise AssertionError("expected ValueError")
