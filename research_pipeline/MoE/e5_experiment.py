"""E5 实验：用 E2 路由选择代表因子，再在其完整因子群内等权。

本模块只复用原项目已有的数据处理、E2 路由和回测能力，不修改 E1-E4。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from ..research_system.behavior import build_daily_factor_performance
from ..research_system.clustering import fit_behavior_kmeans, select_representatives
from ..research_system.experiments import (
    _behavior_frame,
    _build_alpha,
    _check_snapshot,
    _fit_router,
    _market_feature_frame,
    _performance_tensor,
    _rolling_table,
    _route_weights,
    _target_tensor,
)
from ..research_system.features import make_forward_return
from ..research_system.io import resolve_snapshot, write_json
from ..research_system.portfolio import backtest_long_only


def build_factor_group_table(
    features: pd.DataFrame,
    labels: np.ndarray | list[int],
    centers: np.ndarray,
) -> pd.DataFrame:
    """保存全部聚类成员，并将离中心最近的因子标记为代表因子。"""
    labels_array = np.asarray(labels, dtype=int)
    centers_array = np.asarray(centers, dtype=float)
    if len(features) != len(labels_array):
        raise ValueError("labels and features must have the same row count")

    numeric_columns = [
        column
        for column in features.columns
        if column not in {"factor_id", "date", "as_of_date"}
        and pd.api.types.is_numeric_dtype(features[column])
    ]
    if not numeric_columns:
        raise ValueError("features has no numeric feature columns")
    if centers_array.ndim != 2 or centers_array.shape[1] != len(numeric_columns):
        raise ValueError("centers do not match numeric feature columns")

    matrix = features[numeric_columns].apply(pd.to_numeric, errors="coerce")
    matrix = matrix.replace([np.inf, -np.inf], np.nan)
    matrix = matrix.fillna(matrix.median()).fillna(0.0)
    factor_ids = (
        features["factor_id"].astype(str).tolist()
        if "factor_id" in features.columns
        else [str(item) for item in features.index]
    )
    representatives = select_representatives(
        labels_array, features, centers_array, per_cluster=1
    )
    representative_by_cluster: dict[int, str] = {}
    factor_to_cluster = dict(zip(factor_ids, labels_array.tolist()))
    for factor_id in representatives:
        representative_by_cluster[int(factor_to_cluster[factor_id])] = factor_id

    rows: list[dict[str, Any]] = []
    for position, factor_id in enumerate(factor_ids):
        cluster = int(labels_array[position])
        distance = float(
            np.linalg.norm(
                matrix.iloc[position].to_numpy(dtype=float) - centers_array[cluster]
            )
        )
        representative = representative_by_cluster[cluster]
        rows.append(
            {
                "cluster": cluster,
                "factor_id": factor_id,
                "representative_factor_id": representative,
                "is_representative": factor_id == representative,
                "distance_to_center": distance,
            }
        )

    result = pd.DataFrame(rows)
    group_sizes = result.groupby("cluster")["factor_id"].transform("size")
    result["group_size"] = group_sizes.astype(int)
    return result.sort_values(
        ["cluster", "distance_to_center", "factor_id"], kind="stable"
    ).reset_index(drop=True)


def select_top_representative_group(
    representative_weights: pd.DataFrame,
    factor_groups: pd.DataFrame,
) -> pd.DataFrame:
    """按日期选择 E2 权重最高的代表因子及其所属因子群。"""
    required_weights = {"date", "factor_id", "weight"}
    required_groups = {
        "cluster",
        "factor_id",
        "representative_factor_id",
        "is_representative",
        "group_size",
    }
    if missing := required_weights - set(representative_weights.columns):
        raise ValueError(f"representative_weights missing columns: {sorted(missing)}")
    if missing := required_groups - set(factor_groups.columns):
        raise ValueError(f"factor_groups missing columns: {sorted(missing)}")

    representative_rows = factor_groups[factor_groups["is_representative"]].copy()
    representative_rows = representative_rows[
        ["cluster", "representative_factor_id", "group_size"]
    ].drop_duplicates()
    candidates = representative_weights.rename(
        columns={
            "factor_id": "representative_factor_id",
            "weight": "representative_weight",
        }
    ).merge(
        representative_rows,
        on="representative_factor_id",
        how="inner",
        validate="many_to_one",
    )
    if candidates.empty:
        raise ValueError("no representative weights can be mapped to factor groups")

    # 权重相同时按代表因子编号排序，保证结果可复现。
    selected = candidates.sort_values(
        ["date", "representative_weight", "representative_factor_id"],
        ascending=[True, False, True],
        kind="stable",
    ).drop_duplicates(subset=["date"], keep="first")
    return selected[
        [
            "date",
            "cluster",
            "representative_factor_id",
            "representative_weight",
            "group_size",
        ]
    ].sort_values("date").reset_index(drop=True)


def expand_selected_groups_to_equal_weights(
    selected_groups: pd.DataFrame,
    factor_groups: pd.DataFrame,
) -> pd.DataFrame:
    """把每日选中的完整因子群展开，并对群内全部因子等权。"""
    required_selected = {"date", "cluster", "representative_factor_id"}
    if missing := required_selected - set(selected_groups.columns):
        raise ValueError(f"selected_groups missing columns: {sorted(missing)}")
    if factor_groups.empty:
        raise ValueError("factor_groups cannot be empty")

    members = factor_groups[["cluster", "factor_id"]].drop_duplicates()
    expanded = selected_groups.merge(
        members, on="cluster", how="left", validate="many_to_many"
    )
    expanded["group_size"] = expanded.groupby("date")["factor_id"].transform("size")
    expanded["weight"] = 1.0 / expanded["group_size"].astype(float)
    return expanded[
        [
            "date",
            "cluster",
            "representative_factor_id",
            "factor_id",
            "group_size",
            "weight",
        ]
    ].sort_values(["date", "factor_id"]).reset_index(drop=True)


def _save_factor_groups(run_dir: Path, factor_groups: pd.DataFrame) -> None:
    """同时保存总表及每个簇的独立成员清单。"""
    factor_groups.to_parquet(run_dir / "factor_groups.parquet", index=False)
    groups_dir = run_dir / "factor_groups"
    groups_dir.mkdir()
    all_groups: list[dict[str, Any]] = []
    for cluster, group in factor_groups.groupby("cluster", sort=True):
        representative = str(group["representative_factor_id"].iloc[0])
        payload = {
            "cluster": int(cluster),
            "representative_factor_id": representative,
            "group_size": int(len(group)),
            "factor_ids": group["factor_id"].astype(str).tolist(),
        }
        write_json(groups_dir / f"cluster_{int(cluster):03d}.json", payload)
        all_groups.append(payload)
    write_json(run_dir / "factor_groups.json", {"groups": all_groups})


def run_e5_experiment(
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
    top_n: int = 20,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.001,
    rolling_window: int = 40,
    epochs: int = 10,
    random_state: int = 42,
) -> dict[str, Any]:
    """运行独立 E5，并保存聚类、路由、等权信号和回测产物。"""
    snapshot = resolve_snapshot(data_root, snapshot_id)
    values_root = Path(values_run).expanduser().resolve() / "factor_values"
    state_path = Path(state_run).expanduser().resolve() / "market_states.parquet"
    _check_snapshot(
        values_root.parent, "factor_values_manifest.json", snapshot.name
    )
    _check_snapshot(state_path.parent, "manifest.json", snapshot.name)
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
    selected_ids = [
        factor_id for factor_id in available if factor_id in requested_set
    ]
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
    candidate_values = pd.concat(values_parts, ignore_index=True)
    all_dates = pd.DatetimeIndex(
        pd.to_datetime(market["date"], errors="coerce").dropna().unique()
    ).sort_values()
    if test_start is None:
        test_start_ts = all_dates[max(1, int(len(all_dates) * 0.8))]
    else:
        test_start_ts = pd.Timestamp(test_start)
    train_end = test_start_ts

    behavior = _behavior_frame(performance, selected_ids, train_end)
    cluster_count = min(n_clusters, len(selected_ids))
    clustering = fit_behavior_kmeans(
        behavior, n_clusters=cluster_count, random_state=random_state
    )
    factor_groups = build_factor_group_table(
        behavior, clustering["labels"], clustering["centers"]
    )
    representatives = (
        factor_groups[factor_groups["is_representative"]]
        .sort_values("cluster")["factor_id"]
        .astype(str)
        .tolist()
    )
    if not representatives:
        raise ValueError("K-Means did not produce any representative factor")

    rolling = _rolling_table(
        performance, representatives, all_dates, rolling_window
    )
    market_features = _market_feature_frame(states, all_dates)
    performance_input = _performance_tensor(
        rolling, all_dates, representatives
    )
    target_tensor = _target_tensor(performance, all_dates, representatives)
    train_mask = all_dates < train_end

    # 完全复用 E2：只使用市场特征训练代表因子路由，不使用绩效路由。
    model = _fit_router(
        market_features,
        performance_input,
        target_tensor,
        train_mask,
        len(representatives),
        epochs,
        random_state,
        use_performance=False,
        use_turnover_penalty=False,
        factor_top_k=len(representatives),
    )
    representative_weights = _route_weights(
        model,
        market_features,
        performance_input,
        all_dates,
        representatives,
        route="market",
        top_k=len(representatives),
    )
    selected_groups = select_top_representative_group(
        representative_weights, factor_groups
    )
    factor_weights = expand_selected_groups_to_equal_weights(
        selected_groups, factor_groups
    )

    output_base = Path(output_root).expanduser().resolve()
    seed = json.dumps(
        {
            "snapshot": snapshot.name,
            "factors": selected_ids,
            "test_start": str(test_start_ts),
            "random_state": random_state,
            "experiment_id": "E5",
        },
        sort_keys=True,
    )
    run_id = (
        f"e5_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_"
        f"{hashlib.sha256(seed.encode()).hexdigest()[:8]}"
    )
    run_dir = output_base / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    performance.to_parquet(
        run_dir / "daily_factor_performance.parquet", index=False
    )
    behavior.reset_index().to_parquet(
        run_dir / "behavior_features.parquet", index=False
    )
    write_json(
        run_dir / "clustering.json",
        {
            "n_clusters": cluster_count,
            "feature_columns": clustering["feature_columns"],
            "centers": clustering["centers"].tolist(),
            "inertia": clustering["inertia"],
            "random_state": random_state,
            "representatives": representatives,
            "training_end": str(train_end),
        },
    )
    _save_factor_groups(run_dir, factor_groups)
    representative_weights.to_parquet(
        run_dir / "representative_weights.parquet", index=False
    )
    selected_groups.to_parquet(
        run_dir / "selected_groups.parquet", index=False
    )
    factor_weights.to_parquet(
        run_dir / "factor_weights.parquet", index=False
    )
    torch.save(model.state_dict(), run_dir / "model.pt")

    test_weights = factor_weights[
        pd.to_datetime(factor_weights["date"]).ge(test_start_ts)
    ]
    alpha = _build_alpha(candidate_values, test_weights)
    alpha.to_parquet(run_dir / "alpha.parquet", index=False)
    backtest = backtest_long_only(
        market,
        alpha,
        top_n=top_n,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
    )
    backtest["daily"].to_parquet(run_dir / "daily.parquet", index=False)
    write_json(run_dir / "metrics.json", backtest["metrics"])

    config = {
        "experiment_id": "E5",
        "name": "E2 top-representative group equal-weight",
        "components": [
            "kmeans_factor_groups",
            "macoe",
            "top_representative_group",
            "group_equal_weight",
            "cost_backtest",
        ],
        "snapshot_id": snapshot.name,
        "factor_ids": selected_ids,
        "representatives": representatives,
        "test_start": str(test_start_ts),
        "training_end": str(train_end),
        "horizon": horizon,
        "n_clusters": cluster_count,
        "top_n": top_n,
        "fee_rate": fee_rate,
        "slippage_rate": slippage_rate,
        "rolling_window": rolling_window,
        "epochs": epochs,
        "random_state": random_state,
    }
    write_json(run_dir / "config.json", config)

    manifest = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "experiment_id": "E5",
        "snapshot_id": snapshot.name,
        "selected_factor_count": len(selected_ids),
        "cluster_count": cluster_count,
        "representative_factor_count": len(representatives),
        "selected_factor_ids": selected_ids,
        "representatives": representatives,
        "test_start": str(test_start_ts),
        "metrics": backtest["metrics"],
        "files": [
            "daily_factor_performance.parquet",
            "behavior_features.parquet",
            "clustering.json",
            "factor_groups.parquet",
            "factor_groups.json",
            "representative_weights.parquet",
            "selected_groups.parquet",
            "factor_weights.parquet",
            "model.pt",
            "alpha.parquet",
            "daily.parquet",
            "metrics.json",
            "config.json",
        ],
    }
    write_json(run_dir / "e5_manifest.json", manifest)
    return manifest
