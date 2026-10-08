"""在训练期行为画像上执行 K-Means，并保存完整因子群。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ..common.context import prepare_experiment_context
from ..common.io import write_json
from .clustering import drop_empty_behavior_factors, fit_behavior_kmeans
from .grouping import build_factor_group_table, save_factor_groups


def run_cluster(
    data_root: str | Path,
    values_run: str | Path,
    state_run: str | Path,
    output_root: str | Path,
    snapshot_id: str | None = None,
    factor_ids: Iterable[str] | None = None,
    max_factors: int | None = None,
    test_start: object | None = None,
    horizon: int = 20,
    n_clusters: int = 20,
    random_state: int = 42,
) -> dict[str, Any]:
    """生成一份供 E1–E6 共用的聚类产物。"""
    context = prepare_experiment_context(
        data_root=data_root,
        values_run=values_run,
        state_run=state_run,
        snapshot_id=snapshot_id,
        factor_ids=factor_ids,
        max_factors=max_factors,
        test_start=test_start,
        horizon=horizon,
    )
    # 训练期画像全空的因子不进聚类，但仍保留在日度绩效表里便于核对。
    usable_behavior, skipped_ids = drop_empty_behavior_factors(context.behavior)
    usable_ids = [str(item) for item in usable_behavior.index]
    if usable_behavior.empty:
        skipped_text = ", ".join(skipped_ids) if skipped_ids else "(none)"
        raise ValueError(
            "each factor needs at least one finite behavior feature; "
            f"skipped: {skipped_text}"
        )
    cluster_count = min(n_clusters, len(usable_ids))
    clustering = fit_behavior_kmeans(
        usable_behavior,
        n_clusters=cluster_count,
        random_state=random_state,
    )
    factor_groups = build_factor_group_table(
        usable_behavior,
        clustering["labels"],
        clustering["centers"],
    )
    representatives = (
        factor_groups[factor_groups["is_representative"]]
        .sort_values("cluster")["factor_id"]
        .astype(str)
        .tolist()
    )
    if not representatives:
        representatives = usable_ids[:1]

    seed = json.dumps(
        {
            "snapshot": context.snapshot.name,
            "factors": usable_ids,
            "test_start": str(context.test_start),
            "random_state": random_state,
            "n_clusters": cluster_count,
        },
        sort_keys=True,
    )
    run_id = (
        f"cluster_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_"
        f"{hashlib.sha256(seed.encode()).hexdigest()[:8]}"
    )
    run_dir = Path(output_root).expanduser().resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    context.performance.to_parquet(
        run_dir / "daily_factor_performance.parquet",
        index=False,
    )
    usable_behavior.reset_index().to_parquet(
        run_dir / "behavior_features.parquet",
        index=False,
    )
    pd.DataFrame(
        {"factor_id": usable_ids, "cluster": clustering["labels"]}
    ).to_parquet(run_dir / "factor_clusters.parquet", index=False)
    write_json(
        run_dir / "clustering.json",
        {
            "n_clusters": int(cluster_count),
            "feature_columns": clustering["feature_columns"],
            "centers": clustering["centers"].tolist(),
            "inertia": clustering["inertia"],
            "random_state": random_state,
            "representatives": representatives,
            "training_end": str(context.train_end),
        },
    )
    save_factor_groups(run_dir, factor_groups)

    manifest = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "snapshot_id": context.snapshot.name,
        "data_root": str(Path(data_root).expanduser().resolve()),
        "values_run": str(Path(values_run).expanduser().resolve()),
        "state_run": str(Path(state_run).expanduser().resolve()),
        "selected_factor_count": len(usable_ids),
        "skipped_factor_count": len(skipped_ids),
        "cluster_count": cluster_count,
        "representative_factor_count": len(representatives),
        "selected_factor_ids": usable_ids,
        "skipped_factor_ids": skipped_ids,
        "representatives": representatives,
        "test_start": str(context.test_start),
        "training_end": str(context.train_end),
        "horizon": horizon,
        "n_clusters": cluster_count,
        "random_state": random_state,
        "files": [
            "daily_factor_performance.parquet",
            "behavior_features.parquet",
            "factor_clusters.parquet",
            "clustering.json",
            "factor_groups.parquet",
            "factor_groups.json",
        ],
    }
    write_json(run_dir / "cluster_manifest.json", manifest)
    return manifest
