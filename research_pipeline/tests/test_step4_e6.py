import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from research_pipeline.step3_clustering.cluster import run_cluster
from research_pipeline.step3_clustering.grouping import (
    apply_min_stay_primary_cluster,
    blend_inner_weights_with_group_weights,
    merge_small_clusters,
    select_active_inner_weights,
    select_soft_top2_groups,
)
from research_pipeline.step4_experiments.routing import fit_router, route_weights
from research_pipeline.step4_experiments.run import run_experiments
from research_pipeline.tests.helpers import write_experiment_inputs


def test_select_active_inner_weights_uses_only_outer_selected_group():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    selected_groups = pd.DataFrame(
        [
            {
                "date": dates[0],
                "cluster": 0,
                "representative_factor_id": "f1",
            },
            {
                "date": dates[1],
                "cluster": 1,
                "representative_factor_id": "f3",
            },
        ]
    )
    inner_weights = pd.DataFrame(
        [
            {"date": dates[0], "cluster": 0, "factor_id": "f1", "weight": 0.8},
            {"date": dates[0], "cluster": 0, "factor_id": "f2", "weight": 0.2},
            {"date": dates[0], "cluster": 1, "factor_id": "f3", "weight": 0.3},
            {"date": dates[0], "cluster": 1, "factor_id": "f4", "weight": 0.7},
            {"date": dates[1], "cluster": 0, "factor_id": "f1", "weight": 0.4},
            {"date": dates[1], "cluster": 0, "factor_id": "f2", "weight": 0.6},
            {"date": dates[1], "cluster": 1, "factor_id": "f3", "weight": 0.25},
            {"date": dates[1], "cluster": 1, "factor_id": "f4", "weight": 0.75},
        ]
    )

    result = select_active_inner_weights(selected_groups, inner_weights)

    assert result.groupby("date")["factor_id"].apply(list).tolist() == [
        ["f1", "f2"],
        ["f3", "f4"],
    ]
    assert result.groupby("date")["weight"].sum().eq(1.0).all()
    assert result["weight"].tolist() == [0.8, 0.2, 0.25, 0.75]


def test_run_e6_writes_double_router_outputs(tmp_path):
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
        experiments="E6",
        top_n=1,
        epochs=1,
    )

    e6 = result["e6"]
    run_dir = Path(e6["run_dir"])
    assert e6["experiment_id"] == "E6"
    assert e6["inner_router_count"] == 1
    assert (run_dir / "models" / "outer_router.pt").exists()
    assert (run_dir / "models" / "inner" / "cluster_000.pt").exists()
    assert (run_dir / "outer_representative_weights.parquet").exists()
    assert (run_dir / "inner_factor_weights.parquet").exists()
    assert (run_dir / "factor_weights.parquet").exists()
    assert (run_dir / "inner_router_configs.json").exists()
    assert (run_dir / "metrics.json").exists()

    factor_weights = pd.read_parquet(run_dir / "factor_weights.parquet")
    selected_groups = pd.read_parquet(run_dir / "selected_groups.parquet")
    config_text = (run_dir / "config.json").read_text(encoding="utf-8")
    config = json.loads(config_text)
    assert factor_weights.groupby("date")["factor_id"].nunique().eq(2).all()
    assert factor_weights.groupby("date")["weight"].sum().round(7).eq(1.0).all()
    assert selected_groups.groupby("date").size().eq(1).all()
    assert selected_groups["group_weight"].eq(1.0).all()
    assert '"min_stay_days": 20' in config_text
    assert '"soft_top_k": 1' in config_text
    assert "min_stay_primary_group" in config_text
    assert "soft_top2_group" not in config_text
    assert config["n_clusters"] == 1
    assert config["min_cluster_size"] == 3
    assert config["load_balance_coef"] == 0.01
    assert config["cluster_merge_map"] == {"0": 0}
    assert "merge_small_clusters" in config["components"]
    assert "outer_load_balance" in config["components"]


def test_min_stay_primary_cluster_waits_before_switching():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    ranked = pd.DataFrame(
        [
            {
                "date": dates[0],
                "cluster": 0,
                "representative_factor_id": "f1",
                "representative_weight": 0.8,
                "group_size": 2,
            },
            {
                "date": dates[0],
                "cluster": 1,
                "representative_factor_id": "f3",
                "representative_weight": 0.2,
                "group_size": 2,
            },
            {
                "date": dates[1],
                "cluster": 0,
                "representative_factor_id": "f1",
                "representative_weight": 0.1,
                "group_size": 2,
            },
            {
                "date": dates[1],
                "cluster": 1,
                "representative_factor_id": "f3",
                "representative_weight": 0.9,
                "group_size": 2,
            },
            {
                "date": dates[2],
                "cluster": 0,
                "representative_factor_id": "f1",
                "representative_weight": 0.1,
                "group_size": 2,
            },
            {
                "date": dates[2],
                "cluster": 1,
                "representative_factor_id": "f3",
                "representative_weight": 0.9,
                "group_size": 2,
            },
        ]
    )

    result = apply_min_stay_primary_cluster(ranked, min_stay=2)

    assert result["cluster"].tolist() == [0, 0, 1]


def test_blend_inner_weights_mixes_primary_and_secondary_groups():
    dates = pd.to_datetime(["2024-01-01"])
    selected_groups = pd.DataFrame(
        [
            {
                "date": dates[0],
                "cluster": 0,
                "representative_factor_id": "f1",
                "group_weight": 0.7,
            },
            {
                "date": dates[0],
                "cluster": 1,
                "representative_factor_id": "f3",
                "group_weight": 0.3,
            },
        ]
    )
    inner_weights = pd.DataFrame(
        [
            {"date": dates[0], "cluster": 0, "factor_id": "f1", "weight": 0.8},
            {"date": dates[0], "cluster": 0, "factor_id": "f2", "weight": 0.2},
            {"date": dates[0], "cluster": 1, "factor_id": "f3", "weight": 0.25},
            {"date": dates[0], "cluster": 1, "factor_id": "f4", "weight": 0.75},
        ]
    )

    result = blend_inner_weights_with_group_weights(selected_groups, inner_weights)

    assert result["factor_id"].tolist() == ["f1", "f2", "f3", "f4"]
    assert result["weight"].tolist() == pytest.approx([0.56, 0.14, 0.075, 0.225])
    assert result["weight"].sum() == pytest.approx(1.0)


def test_select_soft_top2_groups_degenerates_to_single_cluster():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    groups = pd.DataFrame(
        [
            {
                "cluster": 0,
                "factor_id": "f1",
                "representative_factor_id": "f1",
                "is_representative": True,
                "group_size": 2,
            },
            {
                "cluster": 0,
                "factor_id": "f2",
                "representative_factor_id": "f1",
                "is_representative": False,
                "group_size": 2,
            },
        ]
    )
    weights = pd.DataFrame(
        [
            {"date": dates[0], "factor_id": "f1", "weight": 1.0},
            {"date": dates[1], "factor_id": "f1", "weight": 1.0},
        ]
    )

    result = select_soft_top2_groups(weights, groups, min_stay=20)

    assert result.groupby("date").size().eq(1).all()
    assert result["group_weight"].eq(1.0).all()
    assert result["is_primary"].all()
    assert set(result["cluster"].astype(int)) == {0}


def test_select_soft_top2_groups_renormalizes_outer_weights():
    dates = pd.to_datetime(["2024-01-01"])
    groups = pd.DataFrame(
        [
            {
                "cluster": 0,
                "factor_id": "f1",
                "representative_factor_id": "f1",
                "is_representative": True,
                "group_size": 2,
            },
            {
                "cluster": 0,
                "factor_id": "f2",
                "representative_factor_id": "f1",
                "is_representative": False,
                "group_size": 2,
            },
            {
                "cluster": 1,
                "factor_id": "f3",
                "representative_factor_id": "f3",
                "is_representative": True,
                "group_size": 2,
            },
            {
                "cluster": 1,
                "factor_id": "f4",
                "representative_factor_id": "f3",
                "is_representative": False,
                "group_size": 2,
            },
        ]
    )
    weights = pd.DataFrame(
        [
            {"date": dates[0], "factor_id": "f1", "weight": 0.7},
            {"date": dates[0], "factor_id": "f3", "weight": 0.3},
        ]
    )

    result = select_soft_top2_groups(weights, groups, min_stay=1, soft_top_k=2)

    assert result["cluster"].tolist() == [0, 1]
    assert result["is_primary"].tolist() == [True, False]
    assert result["group_weight"].tolist() == pytest.approx([0.7, 0.3])

    primary_only = select_soft_top2_groups(weights, groups, min_stay=1, soft_top_k=1)
    assert len(primary_only) == 1
    assert int(primary_only.iloc[0]["cluster"]) == 0
    assert float(primary_only.iloc[0]["group_weight"]) == 1.0
    assert bool(primary_only.iloc[0]["is_primary"])


def test_merge_small_clusters_absorbs_into_nearest_large():
    groups = pd.DataFrame(
        [
            {
                "cluster": 0,
                "factor_id": "a",
                "representative_factor_id": "a",
                "is_representative": True,
                "group_size": 1,
            },
            {
                "cluster": 1,
                "factor_id": "b",
                "representative_factor_id": "b",
                "is_representative": True,
                "group_size": 1,
            },
            {
                "cluster": 2,
                "factor_id": "c",
                "representative_factor_id": "c",
                "is_representative": True,
                "group_size": 4,
            },
            {
                "cluster": 2,
                "factor_id": "d",
                "representative_factor_id": "c",
                "is_representative": False,
                "group_size": 4,
            },
            {
                "cluster": 2,
                "factor_id": "e",
                "representative_factor_id": "c",
                "is_representative": False,
                "group_size": 4,
            },
            {
                "cluster": 2,
                "factor_id": "f",
                "representative_factor_id": "c",
                "is_representative": False,
                "group_size": 4,
            },
        ]
    )
    centers = np.array(
        [
            [0.05, 0.0],
            [0.08, 0.0],
            [0.0, 0.0],
        ],
        dtype=float,
    )

    merged, merge_map = merge_small_clusters(groups, centers, min_size=3)

    assert merged["cluster"].nunique() == 1
    assert set(merged["factor_id"]) == {"a", "b", "c", "d", "e", "f"}
    assert set(merged["representative_factor_id"]) == {"c"}
    assert merged["is_representative"].sum() == 1
    assert merged.loc[merged["is_representative"], "factor_id"].iloc[0] == "c"
    assert int(merged["group_size"].iloc[0]) == 6
    assert merge_map == {0: 0, 1: 0, 2: 0}


def test_merge_small_clusters_all_small_reaches_min_size_or_one():
    groups = pd.DataFrame(
        [
            {
                "cluster": cluster,
                "factor_id": f"f{cluster}_{index}",
                "representative_factor_id": f"f{cluster}_0",
                "is_representative": index == 0,
                "group_size": 2,
            }
            for cluster in (0, 1, 2)
            for index in (0, 1)
        ]
    )
    centers = np.array([[0.0], [0.1], [10.0]], dtype=float)

    merged, merge_map = merge_small_clusters(groups, centers, min_size=3)

    sizes = merged.groupby("cluster")["factor_id"].nunique()
    assert len(sizes) >= 1
    assert bool((sizes.ge(3).all()) or (len(sizes) == 1))
    assert set(merge_map) == {0, 1, 2}
    assert merged["factor_id"].nunique() == 6


def test_fit_router_with_load_balance_coef_keeps_positive_weights():
    dates = pd.date_range("2024-01-01", periods=6)
    market = pd.DataFrame(
        {
            "market_return": [0.01] * 6,
            "market_volatility": [0.02] * 6,
            "up_ratio": [0.5] * 6,
            "cross_section_dispersion": [0.1] * 6,
        }
    )
    factor_count = 3
    performance = torch.zeros((6, factor_count, 9), dtype=torch.float32)
    target = torch.tensor(
        [
            [0.2, 0.1, -0.1],
            [0.1, 0.2, -0.05],
            [0.15, 0.05, 0.0],
            [0.0, 0.1, 0.2],
            [-0.1, 0.05, 0.15],
            [0.05, 0.0, 0.1],
        ],
        dtype=torch.float32,
    )
    train_mask = np.array([True, True, True, True, False, False])
    model = fit_router(
        market,
        performance,
        target,
        train_mask,
        factor_count,
        epochs=2,
        random_state=0,
        use_performance=False,
        use_turnover_penalty=False,
        factor_top_k=factor_count,
        load_balance_coef=0.01,
    )
    weights = route_weights(
        model,
        market,
        performance,
        dates,
        ["r0", "r1", "r2"],
        route="market",
        top_k=factor_count,
    )

    assert weights["weight"].gt(0).all()
    assert weights.groupby("date")["weight"].sum().round(6).eq(1.0).all()
