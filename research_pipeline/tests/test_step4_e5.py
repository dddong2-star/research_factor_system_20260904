from pathlib import Path
import json

import numpy as np
import pandas as pd

from research_pipeline.step3_clustering.cluster import run_cluster
from research_pipeline.step3_clustering.grouping import (
    build_factor_group_table,
    expand_selected_groups_to_equal_weights,
    select_top_representative_group,
)
from research_pipeline.step4_experiments.run import run_experiments
from research_pipeline.tests.helpers import write_experiment_inputs


def test_e5_selects_top_representative_group_and_equal_weights_members():
    features = pd.DataFrame(
        {"feature": [0.0, 3.0, 10.0, 15.0]},
        index=["f1", "f2", "f3", "f4"],
    )
    groups = build_factor_group_table(
        features,
        labels=np.array([0, 0, 1, 1]),
        centers=np.array([[1.0], [11.0]]),
    )

    assert groups.groupby("cluster")["factor_id"].apply(list).to_dict() == {
        0: ["f1", "f2"],
        1: ["f3", "f4"],
    }
    assert groups[groups["is_representative"]]["factor_id"].tolist() == [
        "f1",
        "f3",
    ]

    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    representative_weights = pd.DataFrame(
        [
            {"date": dates[0], "factor_id": "f1", "weight": 0.8},
            {"date": dates[0], "factor_id": "f3", "weight": 0.2},
            {"date": dates[1], "factor_id": "f1", "weight": 0.1},
            {"date": dates[1], "factor_id": "f3", "weight": 0.9},
        ]
    )
    selected = select_top_representative_group(representative_weights, groups)
    weights = expand_selected_groups_to_equal_weights(selected, groups)

    assert selected["representative_factor_id"].tolist() == ["f1", "f3"]
    assert weights.groupby("date")["factor_id"].apply(list).tolist() == [
        ["f1", "f2"],
        ["f3", "f4"],
    ]
    assert weights.groupby("date")["weight"].sum().eq(1.0).all()
    assert weights["weight"].eq(0.5).all()


def test_run_e5_writes_group_and_backtest_outputs(tmp_path):
    inputs = write_experiment_inputs(tmp_path)
    cluster = run_cluster(
        data_root=inputs["data_root"],
        values_run=inputs["values_run"],
        state_run=inputs["state_run"],
        output_root=tmp_path / "clusters",
        snapshot_id="snap",
        test_start="2024-01-05",
        horizon=1,
        n_clusters=1,
    )

    result = run_experiments(
        cluster_run=cluster["run_dir"],
        output_root=tmp_path / "runs",
        experiments="E5",
        top_n=1,
        epochs=1,
    )

    e5 = result["e5"]
    run_dir = Path(e5["run_dir"])
    assert e5["experiment_id"] == "E5"
    assert e5["selected_factor_count"] == 2
    assert e5["representative_factor_count"] == 1
    assert (run_dir / "factor_groups" / "cluster_000.json").exists()
    assert (run_dir / "representative_weights.parquet").exists()
    assert (run_dir / "selected_groups.parquet").exists()
    assert (run_dir / "factor_weights.parquet").exists()
    assert (run_dir / "model.pt").exists()
    assert (run_dir / "metrics.json").exists()

    factor_weights = pd.read_parquet(run_dir / "factor_weights.parquet")
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert factor_weights.groupby("date")["factor_id"].nunique().eq(2).all()
    assert factor_weights["weight"].eq(0.5).all()
    assert config["n_clusters"] == 1
    assert config["min_cluster_size"] == 3
    assert config["load_balance_coef"] == 0.01
    assert config["cluster_merge_map"] == {"0": 0}
    assert "merge_small_clusters" in config["components"]
    assert "outer_load_balance" in config["components"]
