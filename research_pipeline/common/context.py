"""加载快照、因子值、市场状态和训练期行为画像。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .behavior import build_daily_factor_performance
from .features import make_forward_return
from .io import resolve_snapshot


@dataclass(frozen=True)
class ExperimentContext:
    """一次实验共享且不可变的输入上下文。"""

    snapshot: Path
    market: pd.DataFrame
    states: pd.DataFrame
    selected_ids: list[str]
    factor_values: pd.DataFrame
    performance: pd.DataFrame
    all_dates: pd.DatetimeIndex
    test_start: pd.Timestamp
    train_end: pd.Timestamp
    behavior: pd.DataFrame


def check_snapshot(
    run_dir: Path,
    manifest_name: str,
    snapshot_id: str,
) -> None:
    """确认派生运行目录与请求的不可变快照一致。"""
    manifest_path = run_dir / manifest_name
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_snapshot = str(manifest.get("snapshot_id") or "")
    if run_snapshot and run_snapshot != snapshot_id:
        raise ValueError(
            f"snapshot mismatch: run={run_snapshot!r}, requested={snapshot_id!r}"
        )


def _summary_row(group: pd.DataFrame, factor_id: str) -> dict[str, Any]:
    row: dict[str, Any] = {"factor_id": str(factor_id)}
    for source, target in (
        ("ic", "ic"),
        ("rank_ic", "rank_ic"),
        ("slope", "slope"),
    ):
        values = (
            pd.to_numeric(group[source], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        mean = float(values.mean()) if not values.empty else np.nan
        std = float(values.std(ddof=1)) if len(values) >= 2 else np.nan
        row[f"{target}_mean"] = mean
        row[f"{target}_std"] = std
        row[f"{target}_ir"] = (
            float(mean / std) if np.isfinite(std) and std > 0 else np.nan
        )
    return row


def behavior_frame(
    performance: pd.DataFrame,
    factor_ids: list[str],
    train_end: pd.Timestamp,
) -> pd.DataFrame:
    """仅使用训练边界前已经可获得的标签构造静态行为画像。"""
    mature = performance[
        performance["date"].lt(train_end)
        & performance["label_available_date"].lt(train_end)
    ]
    grouped = {str(key): value for key, value in mature.groupby("factor_id")}
    rows = [
        _summary_row(grouped[factor_id], factor_id)
        if factor_id in grouped
        else {"factor_id": factor_id}
        for factor_id in factor_ids
    ]
    return pd.DataFrame(rows).set_index("factor_id")


def prepare_experiment_context(
    data_root: str | Path,
    values_run: str | Path,
    state_run: str | Path,
    snapshot_id: str | None = None,
    factor_ids: Iterable[str] | None = None,
    max_factors: int | None = None,
    test_start: object | None = None,
    horizon: int = 20,
) -> ExperimentContext:
    """统一准备聚类和实验所需的市场、因子、标签和训练期画像。"""
    snapshot = resolve_snapshot(data_root, snapshot_id)
    values_root = Path(values_run).expanduser().resolve() / "factor_values"
    state_path = Path(state_run).expanduser().resolve() / "market_states.parquet"
    check_snapshot(values_root.parent, "factor_values_manifest.json", snapshot.name)
    check_snapshot(state_path.parent, "manifest.json", snapshot.name)
    if not values_root.exists() or not state_path.exists():
        raise FileNotFoundError(
            "values_run or state_run does not contain required files"
        )

    market = pd.read_parquet(snapshot / "market.parquet")
    states = pd.read_parquet(state_path)
    partitions = sorted(values_root.glob("factor_id=*/values.parquet"))
    available = [path.parent.name.split("=", 1)[-1] for path in partitions]
    requested = (
        [str(item) for item in factor_ids] if factor_ids is not None else available
    )
    requested_set = set(requested)
    selected_ids = [factor_id for factor_id in available if factor_id in requested_set]
    if max_factors is not None:
        if max_factors <= 0:
            raise ValueError("max_factors must be positive")
        selected_ids = selected_ids[:max_factors]
    if not selected_ids:
        raise ValueError("no factor partitions selected")

    target = make_forward_return(market, horizon=horizon)
    performances: list[pd.DataFrame] = []
    values_parts: list[pd.DataFrame] = []
    selected_set = set(selected_ids)
    for path, factor_id in zip(partitions, available):
        if factor_id not in selected_set:
            continue
        values = pd.read_parquet(path, columns=["date", "code", "value"])
        values["factor_id"] = factor_id
        values_parts.append(values)
        performances.append(
            build_daily_factor_performance(values, target, horizon=horizon)
        )

    performance = pd.concat(performances, ignore_index=True)
    factor_values = pd.concat(values_parts, ignore_index=True)
    all_dates = pd.DatetimeIndex(
        pd.to_datetime(market["date"], errors="coerce").dropna().unique()
    ).sort_values()
    if test_start is None:
        test_start_ts = all_dates[max(1, int(len(all_dates) * 0.8))]
    else:
        test_start_ts = pd.Timestamp(test_start)
    train_end = test_start_ts
    behavior = behavior_frame(performance, selected_ids, train_end)

    return ExperimentContext(
        snapshot=snapshot,
        market=market,
        states=states,
        selected_ids=selected_ids,
        factor_values=factor_values,
        performance=performance,
        all_dates=all_dates,
        test_start=test_start_ts,
        train_end=train_end,
        behavior=behavior,
    )
