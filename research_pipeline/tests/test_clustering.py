import numpy as np
import pandas as pd

from research_pipeline.research_system.clustering import (
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
