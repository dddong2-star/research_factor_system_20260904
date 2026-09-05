"""Training-only behavior clustering and representative selection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def _feature_matrix(feature_frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if not isinstance(feature_frame, pd.DataFrame) or feature_frame.empty:
        raise ValueError("feature_frame must be a non-empty DataFrame")
    columns = [
        column for column in feature_frame.columns
        if column not in {"factor_id", "date", "as_of_date"}
        and pd.api.types.is_numeric_dtype(feature_frame[column])
    ]
    if not columns:
        raise ValueError("feature_frame has no numeric feature columns")
    matrix = feature_frame[columns].apply(pd.to_numeric, errors="coerce")
    matrix = matrix.replace([np.inf, -np.inf], np.nan)
    if matrix.isna().all(axis=1).any():
        raise ValueError("each factor needs at least one finite behavior feature")
    matrix = matrix.fillna(matrix.median()).fillna(0.0)
    return matrix, columns


def fit_behavior_kmeans(
    feature_frame: pd.DataFrame,
    n_clusters: int,
    random_state: int = 42,
) -> dict[str, Any]:
    """Fit K-Means on the supplied training feature frame only."""
    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive")
    matrix, columns = _feature_matrix(feature_frame)
    if n_clusters > len(matrix):
        raise ValueError("n_clusters cannot exceed factor row count")
    # Standardization is fitted only on this frame; raw-space centers are
    # returned so representative distances remain interpretable to callers.
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0, ddof=0).replace(0, 1.0)
    standardized = (matrix - mean) / scale
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    labels = model.fit_predict(standardized.to_numpy())
    centers = pd.DataFrame(model.cluster_centers_, columns=columns) * scale + mean
    return {
        "labels": labels,
        "centers": centers.to_numpy(dtype=float),
        "feature_columns": columns,
        "factor_ids": [str(item) for item in feature_frame.index],
        "inertia": float(model.inertia_),
        "random_state": random_state,
    }


def select_representatives(
    labels: np.ndarray | list[int],
    features: pd.DataFrame,
    centers: np.ndarray,
    per_cluster: int = 1,
) -> list[str]:
    """Select nearest factor ids to each fitted cluster center."""
    if per_cluster <= 0:
        raise ValueError("per_cluster must be positive")
    matrix, columns = _feature_matrix(features)
    labels_array = np.asarray(labels)
    centers_array = np.asarray(centers, dtype=float)
    if len(labels_array) != len(matrix):
        raise ValueError("labels and features must have the same row count")
    if centers_array.ndim != 2 or centers_array.shape[1] != len(columns):
        raise ValueError("centers do not match numeric feature columns")
    ids = features["factor_id"].astype(str).tolist() if "factor_id" in features.columns else [str(item) for item in features.index]
    selected: list[str] = []
    for cluster in sorted(set(labels_array.tolist())):
        positions = np.flatnonzero(labels_array == cluster)
        distances = np.linalg.norm(
            matrix.iloc[positions].to_numpy(dtype=float) - centers_array[int(cluster)],
            axis=1,
        )
        order = positions[np.argsort(distances, kind="stable")[:per_cluster]]
        selected.extend(ids[int(position)] for position in order)
    return selected
