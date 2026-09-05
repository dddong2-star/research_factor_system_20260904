import json

import pandas as pd

from research_pipeline.MoE.e6_experiment import (
    run_e6_experiment,
    select_active_inner_weights,
)


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


def test_run_e6_experiment_writes_double_router_outputs(tmp_path):
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
        json.dumps({"snapshot_id": "snap"}),
        encoding="utf-8",
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
        json.dumps({"snapshot_id": "snap"}),
        encoding="utf-8",
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
        json.dumps({"snapshot_id": "snap"}),
        encoding="utf-8",
    )

    result = run_e6_experiment(
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
    assert result["experiment_id"] == "E6"
    assert result["inner_router_count"] == 1
    assert (run_dir / "models" / "outer_router.pt").exists()
    assert (run_dir / "models" / "inner" / "cluster_000.pt").exists()
    assert (run_dir / "outer_representative_weights.parquet").exists()
    assert (run_dir / "inner_factor_weights.parquet").exists()
    assert (run_dir / "factor_weights.parquet").exists()
    assert (run_dir / "inner_router_configs.json").exists()
    assert (run_dir / "metrics.json").exists()

    factor_weights = pd.read_parquet(run_dir / "factor_weights.parquet")
    assert factor_weights.groupby("date")["factor_id"].nunique().eq(2).all()
    assert factor_weights.groupby("date")["weight"].sum().round(7).eq(1.0).all()
