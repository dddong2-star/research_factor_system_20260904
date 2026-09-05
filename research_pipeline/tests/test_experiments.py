from research_pipeline.research_system.experiments import build_experiment_specs
import json

import pandas as pd

from research_pipeline.research_system.experiments import run_experiment_suite


def test_build_experiment_specs_has_incremental_four_level_design():
    specs = build_experiment_specs(["f1", "f2"], test_start="2024-01-03")

    assert [item["experiment_id"] for item in specs] == ["E1", "E2", "E3", "E4"]
    assert specs[0]["components"] == ["kmeans", "equal_weight", "cost_backtest"]
    assert "macoe" in specs[1]["components"]
    assert "pacoe" in specs[2]["components"]
    assert {"macoe", "pacoe", "gumbel_topk", "turnover_penalty"}.issubset(
        specs[3]["components"]
    )
    assert all(item["factor_ids"] == ["f1", "f2"] for item in specs)


def test_run_experiment_suite_writes_four_level_outputs(tmp_path):
    data_root = tmp_path / "research_data"
    snapshot = data_root / "snapshots" / "snap"
    snapshot.mkdir(parents=True)
    dates = pd.date_range("2024-01-01", periods=8)
    market = pd.DataFrame([
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
        for code, close in [("A", 100 + date.day), ("B", 100), ("C", 100 - date.day)]
    ])
    market.to_parquet(snapshot / "market.parquet", index=False)
    (snapshot / "manifest.json").write_text(json.dumps({"snapshot_id": "snap"}), encoding="utf-8")
    value_root = tmp_path / "values" / "factor_values"
    for factor_id, multiplier in [("f1", 1.0), ("f2", -1.0)]:
        partition = value_root / f"factor_id={factor_id}"
        partition.mkdir(parents=True)
        values = market[["date", "code"]].copy()
        values["value"] = values["code"].map({"A": multiplier, "B": 0.0, "C": -multiplier})
        values.to_parquet(partition / "values.parquet", index=False)
    (value_root.parent / "factor_values_manifest.json").write_text(
        json.dumps({"snapshot_id": "snap"}), encoding="utf-8"
    )
    state_root = tmp_path / "states"
    state_root.mkdir()
    pd.DataFrame({
        "date": dates,
        "state_id": ["bull"] * len(dates),
        "market_return": [0.01] * len(dates),
        "market_volatility": [0.01] * len(dates),
        "up_ratio": [0.5] * len(dates),
        "cross_section_dispersion": [0.1] * len(dates),
    }).to_parquet(state_root / "market_states.parquet", index=False)
    (state_root / "manifest.json").write_text(json.dumps({"snapshot_id": "snap"}), encoding="utf-8")

    result = run_experiment_suite(
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
        factor_top_k=1,
    )

    assert [item["experiment_id"] for item in result["experiments"]] == ["E1", "E2", "E3", "E4"]
    assert all((tmp_path / "runs" / result["run_id"] / item["experiment_id"] / "metrics.json").exists() for item in result["experiments"])
    assert (tmp_path / "runs" / result["run_id"] / "daily_factor_performance.parquet").exists()
    assert (tmp_path / "runs" / result["run_id"] / "factor_clusters.parquet").exists()
