from pathlib import Path

import pandas as pd

from research_pipeline.step3_clustering.cluster import run_cluster
from research_pipeline.tests.helpers import write_experiment_inputs


def test_run_cluster_writes_shared_artifacts(tmp_path):
    inputs = write_experiment_inputs(tmp_path)
    result = run_cluster(
        data_root=inputs["data_root"],
        values_run=inputs["values_run"],
        state_run=inputs["state_run"],
        output_root=tmp_path / "clusters",
        snapshot_id="snap",
        test_start="2024-01-05",
        horizon=1,
        n_clusters=1,
    )

    run_dir = Path(result["run_dir"])
    assert result["cluster_count"] == 1
    assert result["representative_factor_count"] == 1
    assert result["skipped_factor_ids"] == []
    assert result["skipped_factor_count"] == 0
    assert (run_dir / "cluster_manifest.json").exists()
    assert (run_dir / "clustering.json").exists()
    assert (run_dir / "factor_groups.parquet").exists()
    assert (run_dir / "factor_groups" / "cluster_000.json").exists()
    assert (run_dir / "daily_factor_performance.parquet").exists()
    assert (run_dir / "behavior_features.parquet").exists()
    assert (run_dir / "factor_clusters.parquet").exists()


def test_run_cluster_skips_factors_without_finite_training_ic(tmp_path):
    inputs = write_experiment_inputs(tmp_path)
    value_root = inputs["values_run"] / "factor_values"
    market = pd.read_parquet(
        inputs["data_root"] / "snapshots" / "snap" / "market.parquet"
    )
    partition = value_root / "factor_id=f_const"
    partition.mkdir(parents=True)
    values = market[["date", "code"]].copy()
    values["value"] = 1.0
    values.to_parquet(partition / "values.parquet", index=False)

    result = run_cluster(
        data_root=inputs["data_root"],
        values_run=inputs["values_run"],
        state_run=inputs["state_run"],
        output_root=tmp_path / "clusters",
        snapshot_id="snap",
        test_start="2024-01-05",
        horizon=1,
        n_clusters=1,
    )

    run_dir = Path(result["run_dir"])
    clusters = pd.read_parquet(run_dir / "factor_clusters.parquet")
    performance = pd.read_parquet(run_dir / "daily_factor_performance.parquet")
    behavior = pd.read_parquet(run_dir / "behavior_features.parquet")

    assert "f_const" in result["skipped_factor_ids"]
    assert result["skipped_factor_count"] == 1
    assert "f_const" not in result["selected_factor_ids"]
    assert set(result["selected_factor_ids"]) == {"f1", "f2"}
    assert "f_const" not in set(clusters["factor_id"].astype(str))
    assert "f_const" not in set(behavior["factor_id"].astype(str))
    assert "f_const" in set(performance["factor_id"].astype(str))
