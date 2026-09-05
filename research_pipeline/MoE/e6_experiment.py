"""E6 双重 MoE：外层选择因子群，内层动态分配群内因子权重。"""

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
from ..research_system.clustering import fit_behavior_kmeans
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
from ..research_system.models import FactorRouter
from ..research_system.portfolio import backtest_long_only
from .e5_experiment import (
    _save_factor_groups,
    build_factor_group_table,
    select_top_representative_group,
)


def select_active_inner_weights(
    selected_groups: pd.DataFrame,
    inner_factor_weights: pd.DataFrame,
) -> pd.DataFrame:
    """按外层每日选群结果，提取对应内层路由的动态因子权重。"""
    required_groups = {"date", "cluster", "representative_factor_id"}
    required_weights = {"date", "cluster", "factor_id", "weight"}
    if missing := required_groups - set(selected_groups.columns):
        raise ValueError(f"selected_groups missing columns: {sorted(missing)}")
    if missing := required_weights - set(inner_factor_weights.columns):
        raise ValueError(f"inner_factor_weights missing columns: {sorted(missing)}")

    selected = selected_groups[
        ["date", "cluster", "representative_factor_id"]
    ].drop_duplicates(["date"])
    active = selected.merge(
        inner_factor_weights,
        on=["date", "cluster"],
        how="left",
        validate="one_to_many",
    )
    if active["factor_id"].isna().any():
        missing_dates = active.loc[active["factor_id"].isna(), "date"].astype(str)
        raise ValueError(
            f"inner router weights are missing for dates: {missing_dates.tolist()}"
        )

    totals = active.groupby("date")["weight"].transform("sum")
    if totals.le(0).any() or ~np.isfinite(totals).all():
        raise ValueError("inner router weights must have a positive finite daily sum")
    active["weight"] = active["weight"] / totals
    return active[
        [
            "date",
            "cluster",
            "representative_factor_id",
            "factor_id",
            "weight",
        ]
    ].sort_values(["date", "factor_id"]).reset_index(drop=True)


def run_e6_experiment(
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
    """运行 E6，并保存两级路由模型、动态权重、Alpha 和回测结果。"""
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
    train_mask = all_dates < train_end

    behavior = _behavior_frame(performance, selected_ids, train_end)
    cluster_count = min(n_clusters, len(selected_ids))
    clustering = fit_behavior_kmeans(
        behavior,
        n_clusters=cluster_count,
        random_state=random_state,
    )
    factor_groups = build_factor_group_table(
        behavior,
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
        raise ValueError("K-Means did not produce any representative factor")

    market_features = _market_feature_frame(states, all_dates)
    rolling = _rolling_table(
        performance,
        selected_ids,
        all_dates,
        rolling_window,
    )

    # 第一级保持 E5/E2 逻辑：只使用市场特征为代表因子评分并硬选择一个群。
    outer_performance_input = _performance_tensor(
        rolling,
        all_dates,
        representatives,
    )
    outer_target = _target_tensor(performance, all_dates, representatives)
    outer_model = _fit_router(
        market_features,
        outer_performance_input,
        outer_target,
        train_mask,
        len(representatives),
        epochs,
        random_state,
        use_performance=False,
        use_turnover_penalty=False,
        factor_top_k=len(representatives),
    )
    outer_weights = _route_weights(
        outer_model,
        market_features,
        outer_performance_input,
        all_dates,
        representatives,
        route="market",
        top_k=len(representatives),
    )
    selected_groups = select_top_representative_group(
        outer_weights,
        factor_groups,
    )

    # 第二级为每个群独立训练一个市场状态 + 滚动绩效融合路由。
    inner_models: dict[int, FactorRouter] = {}
    inner_weight_parts: list[pd.DataFrame] = []
    inner_router_configs: list[dict[str, Any]] = []
    for cluster, group in factor_groups.groupby("cluster", sort=True):
        cluster_id = int(cluster)
        member_ids = group["factor_id"].astype(str).tolist()
        inner_performance_input = _performance_tensor(
            rolling,
            all_dates,
            member_ids,
        )
        inner_target = _target_tensor(performance, all_dates, member_ids)
        inner_seed = random_state + cluster_id + 1
        inner_model = _fit_router(
            market_features,
            inner_performance_input,
            inner_target,
            train_mask,
            len(member_ids),
            epochs,
            inner_seed,
            use_performance=True,
            use_turnover_penalty=False,
            factor_top_k=len(member_ids),
        )
        inner_weights = _route_weights(
            inner_model,
            market_features,
            inner_performance_input,
            all_dates,
            member_ids,
            route="fused",
            top_k=len(member_ids),
        )
        inner_weights["cluster"] = cluster_id
        inner_weight_parts.append(inner_weights)
        inner_models[cluster_id] = inner_model
        inner_router_configs.append(
            {
                "cluster": cluster_id,
                "factor_ids": member_ids,
                "factor_count": len(member_ids),
                "random_state": inner_seed,
                "market_weight_alpha": float(
                    torch.sigmoid(inner_model.alpha_logit).detach()
                ),
            }
        )

    inner_factor_weights = pd.concat(inner_weight_parts, ignore_index=True)
    factor_weights = select_active_inner_weights(
        selected_groups,
        inner_factor_weights,
    )

    output_base = Path(output_root).expanduser().resolve()
    seed = json.dumps(
        {
            "snapshot": snapshot.name,
            "factors": selected_ids,
            "test_start": str(test_start_ts),
            "random_state": random_state,
            "experiment_id": "E6",
        },
        sort_keys=True,
    )
    run_id = (
        f"e6_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_"
        f"{hashlib.sha256(seed.encode()).hexdigest()[:8]}"
    )
    run_dir = output_base / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    performance.to_parquet(
        run_dir / "daily_factor_performance.parquet",
        index=False,
    )
    behavior.reset_index().to_parquet(
        run_dir / "behavior_features.parquet",
        index=False,
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
    outer_weights.to_parquet(
        run_dir / "outer_representative_weights.parquet",
        index=False,
    )
    selected_groups.to_parquet(
        run_dir / "selected_groups.parquet",
        index=False,
    )
    inner_factor_weights.to_parquet(
        run_dir / "inner_factor_weights.parquet",
        index=False,
    )
    factor_weights.to_parquet(
        run_dir / "factor_weights.parquet",
        index=False,
    )

    models_dir = run_dir / "models"
    inner_models_dir = models_dir / "inner"
    inner_models_dir.mkdir(parents=True)
    torch.save(outer_model.state_dict(), models_dir / "outer_router.pt")
    for cluster_id, model in inner_models.items():
        torch.save(
            model.state_dict(),
            inner_models_dir / f"cluster_{cluster_id:03d}.pt",
        )
    write_json(run_dir / "inner_router_configs.json", inner_router_configs)

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
        "experiment_id": "E6",
        "name": "Hierarchical double-router MoE",
        "components": [
            "kmeans_factor_groups",
            "outer_market_router",
            "hard_top1_group",
            "inner_market_performance_router",
            "dense_factor_softmax",
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
        "inner_training_scope": "all_training_dates_per_cluster",
    }
    write_json(run_dir / "config.json", config)

    manifest = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "experiment_id": "E6",
        "snapshot_id": snapshot.name,
        "selected_factor_count": len(selected_ids),
        "cluster_count": cluster_count,
        "representative_factor_count": len(representatives),
        "inner_router_count": len(inner_models),
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
            "outer_representative_weights.parquet",
            "selected_groups.parquet",
            "inner_factor_weights.parquet",
            "factor_weights.parquet",
            "models/outer_router.pt",
            "models/inner/cluster_XXX.pt",
            "inner_router_configs.json",
            "alpha.parquet",
            "daily.parquet",
            "metrics.json",
            "config.json",
        ],
    }
    write_json(run_dir / "e6_manifest.json", manifest)
    return manifest
