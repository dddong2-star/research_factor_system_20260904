import json

import numpy as np
import pandas as pd

from research_pipeline.MoE.e5_experiment import (
    build_factor_group_table,
    expand_selected_groups_to_equal_weights,
    run_e5_experiment,
    select_top_representative_group,
)


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


def test_run_e5_experiment_writes_group_and_backtest_outputs(tmp_path):
    data_root = tmp_path / "research_data"
    snapshot = data_root / "snapshots" / "snap"
    snapshot.mkdir(parents=True)
    dates = pd.date_range("2024-01-01", periods=8)
    market = pd.DataFrame(
        [
            {
                "date": date,
                "code": code,
                "open": close,
                "close": close,
                "high": close,
                "low": close,
                "volume": 100.0,
                "amount": close * 100,
                "Ifsuspend": 0,
            }
            for date in dates
            for code, close in [
                ("A", 100 + date.day),
                ("B", 100),
                ("C", 100 - date.day),
            ]
        ]
    )
    market.to_parquet(snapshot / "market.parquet", index=False)
    (snapshot / "manifest.json").write_text(
        json.dumps({"snapshot_id": "snap"}), encoding="utf-8"
    )

    value_root = tmp_path / "values" / "factor_values"
    for factor_id, multiplier in [("f1", 1.0), ("f2", -1.0)]:
        partition = value_root / f"factor_id={factor_id}"
        partition.mkdir(parents=True)
        values = market[["date", "code"]].copy()
        values["value"] = values["code"].map(
            {"A": multiplier, "B": 0.0, "C": -multiplier}
        )
        values.to_parquet(partition / "values.parquet", index=False)
    (value_root.parent / "factor_values_manifest.json").write_text(
        json.dumps({"snapshot_id": "snap"}), encoding="utf-8"
    )

    state_root = tmp_path / "states"
    state_root.mkdir()
    pd.DataFrame(
        {
            "date": dates,
            "state_id": ["bull"] * len(dates),
            "market_return": [0.01] * len(dates),
            "market_volatility": [0.01] * len(dates),
            "up_ratio": [0.5] * len(dates),
            "cross_section_dispersion": [0.1] * len(dates),
        }
    ).to_parquet(state_root / "market_states.parquet", index=False)
    (state_root / "manifest.json").write_text(
        json.dumps({"snapshot_id": "snap"}), encoding="utf-8"
    )

    result = run_e5_experiment(
        data_root=data_root,
        values_run=value_root.parent,
        state_run=state_root,
        output_root=tmp_path / "runs",
        snapshot_id="snap",
        test_start="2024-01-05",
        horizon=1,
        n_clusters=1,
        top_n=1,
        epochs=1,
    )

    run_dir = tmp_path / "runs" / result["run_id"]
    assert result["experiment_id"] == "E5"
    assert result["selected_factor_count"] == 2
    assert result["representative_factor_count"] == 1
    assert (run_dir / "factor_groups" / "cluster_000.json").exists()
    assert (run_dir / "representative_weights.parquet").exists()
    assert (run_dir / "selected_groups.parquet").exists()
    assert (run_dir / "factor_weights.parquet").exists()
    assert (run_dir / "model.pt").exists()
    assert (run_dir / "metrics.json").exists()

    factor_weights = pd.read_parquet(run_dir / "factor_weights.parquet")
    assert factor_weights.groupby("date")["factor_id"].nunique().eq(2).all()
    assert factor_weights["weight"].eq(0.5).all()
