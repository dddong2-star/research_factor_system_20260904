"""测试用的合成快照、因子值和市场状态。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_experiment_inputs(tmp_path: Path) -> dict[str, Path]:
    """写入与原 E1–E6 测试相同结构的最小输入。"""
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
    return {
        "data_root": data_root,
        "values_run": value_root.parent,
        "state_run": state_root,
    }
