"""根据步骤 3 的聚类产物运行选定的 E1–E6 实验。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch

from ..common.context import ExperimentContext, prepare_experiment_context
from ..common.io import write_json
from ..common.models import FactorRouter
from ..common.portfolio import backtest_long_only
from ..step3_clustering.grouping import (
    E6_MIN_CLUSTER_SIZE,
    E6_MIN_STAY_DAYS,
    E6_OUTER_LOAD_BALANCE_COEF,
    E6_SOFT_TOP_K,
    blend_inner_weights_with_group_weights,
    expand_selected_groups_to_equal_weights,
    merge_small_clusters,
    save_factor_groups,
    select_soft_top2_groups,
    select_top_representative_group,
)
from .alpha import build_alpha
from .routing import (
    fit_router,
    market_feature_frame,
    performance_tensor,
    rolling_table,
    route_weights,
    target_tensor,
)
from .specs import build_experiment_specs

ALL_EXPERIMENTS = ("E1", "E2", "E3", "E4", "E5", "E6")


def parse_experiments(text: str | Iterable[str] | None) -> list[str]:
    """解析实验列表，默认运行全部六级。"""
    if text is None:
        return list(ALL_EXPERIMENTS)
    if isinstance(text, str):
        items = [item.strip().upper() for item in text.replace(" ", "").split(",") if item.strip()]
    else:
        items = [str(item).strip().upper() for item in text]
    if not items or items == ["ALL"]:
        return list(ALL_EXPERIMENTS)
    unknown = [item for item in items if item not in ALL_EXPERIMENTS]
    if unknown:
        raise ValueError(f"unsupported experiments: {unknown}")
    return items


def _load_cluster_manifest(cluster_run: str | Path) -> dict[str, Any]:
    run_dir = Path(cluster_run).expanduser().resolve()
    manifest_path = run_dir / "cluster_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"cluster manifest is missing: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _load_cluster_artifacts(
    cluster_run: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, ExperimentContext]:
    """读取步骤 3 产物，并按其中记录的路径重建实验上下文。"""
    run_dir = Path(cluster_run).expanduser().resolve()
    manifest = _load_cluster_manifest(run_dir)
    clustering = json.loads((run_dir / "clustering.json").read_text(encoding="utf-8"))
    factor_groups = pd.read_parquet(run_dir / "factor_groups.parquet")
    context = prepare_experiment_context(
        data_root=manifest["data_root"],
        values_run=manifest["values_run"],
        state_run=manifest["state_run"],
        snapshot_id=manifest["snapshot_id"],
        factor_ids=manifest["selected_factor_ids"],
        test_start=manifest["test_start"],
        horizon=int(manifest["horizon"]),
    )
    return manifest, clustering, factor_groups, context


def _copy_cluster_files(
    cluster_run: Path,
    target_dir: Path,
    include_groups: bool = False,
) -> None:
    """把步骤 3 的聚类文件复制到实验运行目录，保持原产物字段。"""
    for name in (
        "daily_factor_performance.parquet",
        "behavior_features.parquet",
        "clustering.json",
        "factor_clusters.parquet",
    ):
        source = cluster_run / name
        if source.exists():
            (target_dir / name).write_bytes(source.read_bytes())
    if include_groups:
        save_factor_groups(
            target_dir,
            pd.read_parquet(cluster_run / "factor_groups.parquet"),
        )


def _run_e1_e4(
    context: ExperimentContext,
    clustering: dict[str, Any],
    representatives: list[str],
    selected: list[str],
    output_root: Path,
    cluster_run: Path,
    top_n: int,
    fee_rate: float,
    slippage_rate: float,
    rolling_window: int,
    epochs: int,
    random_state: int,
    factor_top_k: int | None,
) -> dict[str, Any]:
    selected_ids = context.selected_ids
    if factor_top_k is None:
        factor_top_k = len(representatives)
    if factor_top_k <= 0:
        raise ValueError("factor_top_k must be positive")

    candidate_values = context.factor_values[
        context.factor_values["factor_id"].isin(representatives)
    ].copy()
    rolling = rolling_table(
        context.performance, representatives, context.all_dates, rolling_window
    )
    market_features = market_feature_frame(context.states, context.all_dates)
    perf_tensor = performance_tensor(rolling, context.all_dates, representatives)
    targets = target_tensor(context.performance, context.all_dates, representatives)
    train_mask = context.all_dates < context.train_end

    seed = json.dumps(
        {
            "snapshot": context.snapshot.name,
            "factors": representatives,
            "test_start": str(context.test_start),
            "random_state": random_state,
        },
        sort_keys=True,
    )
    run_id = (
        f"experiments_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_"
        f"{hashlib.sha256(seed.encode()).hexdigest()[:8]}"
    )
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _copy_cluster_files(cluster_run, run_dir)
    horizon = int(
        json.loads((cluster_run / "cluster_manifest.json").read_text(encoding="utf-8"))[
            "horizon"
        ]
    )

    specs = [
        spec
        for spec in build_experiment_specs(representatives, context.test_start)
        if spec["experiment_id"] in selected
    ]
    outputs: list[dict[str, Any]] = []
    for spec in specs:
        experiment_id = spec["experiment_id"]
        experiment_dir = run_dir / experiment_id
        experiment_dir.mkdir()
        if experiment_id == "E1":
            weights = route_weights(
                None,
                market_features,
                perf_tensor,
                context.all_dates,
                representatives,
                route="equal",
                top_k=len(representatives),
            )
            model = None
        else:
            use_performance = experiment_id in {"E3", "E4"}
            model = fit_router(
                market_features,
                perf_tensor,
                targets,
                train_mask,
                len(representatives),
                epochs,
                random_state,
                use_performance=use_performance,
                use_turnover_penalty=experiment_id == "E4",
                factor_top_k=min(factor_top_k, len(representatives)),
            )
            route = (
                "market"
                if experiment_id == "E2"
                else "full"
                if experiment_id == "E4"
                else "fused"
            )
            weights = route_weights(
                model,
                market_features,
                perf_tensor,
                context.all_dates,
                representatives,
                route,
                top_k=min(factor_top_k, len(representatives)),
            )
            torch.save(model.state_dict(), experiment_dir / "model.pt")

        weights.to_parquet(experiment_dir / "factor_weights.parquet", index=False)
        test_weights = weights[weights["date"].ge(context.test_start)]
        alpha = build_alpha(candidate_values, test_weights)
        alpha.to_parquet(experiment_dir / "alpha.parquet", index=False)
        backtest = backtest_long_only(
            context.market,
            alpha,
            top_n=top_n,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
        )
        backtest["daily"].to_parquet(experiment_dir / "daily.parquet", index=False)
        write_json(experiment_dir / "metrics.json", backtest["metrics"])
        config = {
            **spec,
            "snapshot_id": context.snapshot.name,
            "horizon": horizon,
            "n_clusters": clustering["n_clusters"],
            "representatives": representatives,
            "fee_rate": fee_rate,
            "slippage_rate": slippage_rate,
            "rolling_window": rolling_window,
            "epochs": epochs,
            "factor_top_k": factor_top_k,
            "random_state": random_state,
        }
        write_json(experiment_dir / "config.json", config)
        outputs.append(
            {
                "experiment_id": experiment_id,
                "components": spec["components"],
                "metrics": backtest["metrics"],
                "directory": str(experiment_dir),
            }
        )
    manifest = {
        "run_id": run_id,
        "snapshot_id": context.snapshot.name,
        "selected_factor_count": len(selected_ids),
        "representative_factor_count": len(representatives),
        "selected_factor_ids": selected_ids,
        "representatives": representatives,
        "test_start": str(context.test_start),
        "training_end": str(context.train_end),
        "horizon": horizon,
        "fee_rate": fee_rate,
        "slippage_rate": slippage_rate,
        "factor_top_k": factor_top_k,
        "random_state": random_state,
        "shared_files": [
            "daily_factor_performance.parquet",
            "behavior_features.parquet",
            "factor_clusters.parquet",
            "clustering.json",
        ],
        "experiments": outputs,
    }
    write_json(run_dir / "experiment_suite.json", manifest)
    return {**manifest, "run_dir": str(run_dir)}


def _run_e5(
    context: ExperimentContext,
    clustering: dict[str, Any],
    factor_groups: pd.DataFrame,
    representatives: list[str],
    output_root: Path,
    cluster_run: Path,
    top_n: int,
    fee_rate: float,
    slippage_rate: float,
    rolling_window: int,
    epochs: int,
    random_state: int,
) -> dict[str, Any]:
    # 只在 E5 运行时合并过小簇，不改写步骤 3 原始产物。
    factor_groups, merge_map = merge_small_clusters(
        factor_groups,
        clustering["centers"],
        min_size=E6_MIN_CLUSTER_SIZE,
    )
    representatives = (
        factor_groups.loc[factor_groups["is_representative"]]
        .sort_values("cluster")["factor_id"]
        .astype(str)
        .tolist()
    )
    if not representatives:
        raise ValueError("merged factor groups have no representatives")
    rolling = rolling_table(
        context.performance, representatives, context.all_dates, rolling_window
    )
    market_features = market_feature_frame(context.states, context.all_dates)
    perf_tensor = performance_tensor(rolling, context.all_dates, representatives)
    targets = target_tensor(context.performance, context.all_dates, representatives)
    train_mask = context.all_dates < context.train_end
    model = fit_router(
        market_features,
        perf_tensor,
        targets,
        train_mask,
        len(representatives),
        epochs,
        random_state,
        use_performance=False,
        use_turnover_penalty=False,
        factor_top_k=len(representatives),
        load_balance_coef=E6_OUTER_LOAD_BALANCE_COEF,
    )
    representative_weights = route_weights(
        model,
        market_features,
        perf_tensor,
        context.all_dates,
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

    seed = json.dumps(
        {
            "snapshot": context.snapshot.name,
            "factors": context.selected_ids,
            "test_start": str(context.test_start),
            "random_state": random_state,
            "experiment_id": "E5",
        },
        sort_keys=True,
    )
    run_id = (
        f"e5_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_"
        f"{hashlib.sha256(seed.encode()).hexdigest()[:8]}"
    )
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _copy_cluster_files(cluster_run, run_dir, include_groups=False)
    save_factor_groups(run_dir, factor_groups)
    representative_weights.to_parquet(
        run_dir / "representative_weights.parquet", index=False
    )
    selected_groups.to_parquet(run_dir / "selected_groups.parquet", index=False)
    factor_weights.to_parquet(run_dir / "factor_weights.parquet", index=False)
    torch.save(model.state_dict(), run_dir / "model.pt")

    test_weights = factor_weights[
        pd.to_datetime(factor_weights["date"]).ge(context.test_start)
    ]
    alpha = build_alpha(context.factor_values, test_weights)
    alpha.to_parquet(run_dir / "alpha.parquet", index=False)
    backtest = backtest_long_only(
        context.market,
        alpha,
        top_n=top_n,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
    )
    backtest["daily"].to_parquet(run_dir / "daily.parquet", index=False)
    write_json(run_dir / "metrics.json", backtest["metrics"])
    cluster_count = int(factor_groups["cluster"].nunique())
    horizon = int(
        json.loads((cluster_run / "cluster_manifest.json").read_text(encoding="utf-8"))[
            "horizon"
        ]
    )
    config = {
        "experiment_id": "E5",
        "name": "E2 top-representative group equal-weight",
        "components": [
            "kmeans_factor_groups",
            "merge_small_clusters",
            "macoe",
            "outer_load_balance",
            "top_representative_group",
            "group_equal_weight",
            "cost_backtest",
        ],
        "snapshot_id": context.snapshot.name,
        "factor_ids": context.selected_ids,
        "representatives": representatives,
        "test_start": str(context.test_start),
        "training_end": str(context.train_end),
        "horizon": horizon,
        "n_clusters": cluster_count,
        "top_n": top_n,
        "fee_rate": fee_rate,
        "slippage_rate": slippage_rate,
        "rolling_window": rolling_window,
        "epochs": epochs,
        "random_state": random_state,
        "min_cluster_size": E6_MIN_CLUSTER_SIZE,
        "load_balance_coef": E6_OUTER_LOAD_BALANCE_COEF,
        "cluster_merge_map": {
            str(source): int(target)
            for source, target in sorted(merge_map.items())
        },
    }
    write_json(run_dir / "config.json", config)
    manifest = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "experiment_id": "E5",
        "snapshot_id": context.snapshot.name,
        "selected_factor_count": len(context.selected_ids),
        "cluster_count": cluster_count,
        "representative_factor_count": len(representatives),
        "selected_factor_ids": context.selected_ids,
        "representatives": representatives,
        "test_start": str(context.test_start),
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


def _run_e6(
    context: ExperimentContext,
    clustering: dict[str, Any],
    factor_groups: pd.DataFrame,
    representatives: list[str],
    output_root: Path,
    cluster_run: Path,
    top_n: int,
    fee_rate: float,
    slippage_rate: float,
    rolling_window: int,
    epochs: int,
    random_state: int,
) -> dict[str, Any]:
    # 只在 E6 运行时合并过小簇，不改写步骤 3 原始产物。
    factor_groups, merge_map = merge_small_clusters(
        factor_groups,
        clustering["centers"],
        min_size=E6_MIN_CLUSTER_SIZE,
    )
    representatives = (
        factor_groups.loc[factor_groups["is_representative"]]
        .sort_values("cluster")["factor_id"]
        .astype(str)
        .tolist()
    )
    if not representatives:
        raise ValueError("merged factor groups have no representatives")
    train_mask = context.all_dates < context.train_end
    market_features = market_feature_frame(context.states, context.all_dates)
    rolling = rolling_table(
        context.performance, context.selected_ids, context.all_dates, rolling_window
    )
    outer_performance_input = performance_tensor(
        rolling, context.all_dates, representatives
    )
    outer_target = target_tensor(
        context.performance, context.all_dates, representatives
    )
    outer_model = fit_router(
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
        load_balance_coef=E6_OUTER_LOAD_BALANCE_COEF,
    )
    outer_weights = route_weights(
        outer_model,
        market_features,
        outer_performance_input,
        context.all_dates,
        representatives,
        route="market",
        top_k=len(representatives),
    )
    selected_groups = select_soft_top2_groups(
        outer_weights,
        factor_groups,
        min_stay=E6_MIN_STAY_DAYS,
        soft_top_k=E6_SOFT_TOP_K,
    )

    inner_models: dict[int, FactorRouter] = {}
    inner_weight_parts: list[pd.DataFrame] = []
    inner_router_configs: list[dict[str, Any]] = []
    for cluster, group in factor_groups.groupby("cluster", sort=True):
        cluster_id = int(cluster)
        member_ids = group["factor_id"].astype(str).tolist()
        inner_performance_input = performance_tensor(
            rolling, context.all_dates, member_ids
        )
        inner_target = target_tensor(context.performance, context.all_dates, member_ids)
        inner_seed = random_state + cluster_id + 1
        inner_model = fit_router(
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
        inner_weights = route_weights(
            inner_model,
            market_features,
            inner_performance_input,
            context.all_dates,
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
    factor_weights = blend_inner_weights_with_group_weights(
        selected_groups,
        inner_factor_weights,
    )

    seed = json.dumps(
        {
            "snapshot": context.snapshot.name,
            "factors": context.selected_ids,
            "test_start": str(context.test_start),
            "random_state": random_state,
            "experiment_id": "E6",
        },
        sort_keys=True,
    )
    run_id = (
        f"e6_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_"
        f"{hashlib.sha256(seed.encode()).hexdigest()[:8]}"
    )
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _copy_cluster_files(cluster_run, run_dir, include_groups=False)
    save_factor_groups(run_dir, factor_groups)
    outer_weights.to_parquet(
        run_dir / "outer_representative_weights.parquet", index=False
    )
    selected_groups.to_parquet(run_dir / "selected_groups.parquet", index=False)
    inner_factor_weights.to_parquet(
        run_dir / "inner_factor_weights.parquet", index=False
    )
    factor_weights.to_parquet(run_dir / "factor_weights.parquet", index=False)

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
        pd.to_datetime(factor_weights["date"]).ge(context.test_start)
    ]
    alpha = build_alpha(context.factor_values, test_weights)
    alpha.to_parquet(run_dir / "alpha.parquet", index=False)
    backtest = backtest_long_only(
        context.market,
        alpha,
        top_n=top_n,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
    )
    backtest["daily"].to_parquet(run_dir / "daily.parquet", index=False)
    write_json(run_dir / "metrics.json", backtest["metrics"])
    cluster_count = int(factor_groups["cluster"].nunique())
    horizon = int(
        json.loads((cluster_run / "cluster_manifest.json").read_text(encoding="utf-8"))[
            "horizon"
        ]
    )
    config = {
        "experiment_id": "E6",
        "name": "Hierarchical double-router MoE",
        "components": [
            "kmeans_factor_groups",
            "merge_small_clusters",
            "outer_market_router",
            "outer_load_balance",
            "min_stay_primary_group",
            "inner_market_performance_router",
            "dense_factor_softmax",
            "cost_backtest",
        ],
        "snapshot_id": context.snapshot.name,
        "factor_ids": context.selected_ids,
        "representatives": representatives,
        "test_start": str(context.test_start),
        "training_end": str(context.train_end),
        "horizon": horizon,
        "n_clusters": cluster_count,
        "top_n": top_n,
        "fee_rate": fee_rate,
        "slippage_rate": slippage_rate,
        "rolling_window": rolling_window,
        "epochs": epochs,
        "random_state": random_state,
        "inner_training_scope": "all_training_dates_per_cluster",
        "min_stay_days": E6_MIN_STAY_DAYS,
        "soft_top_k": E6_SOFT_TOP_K,
        "min_cluster_size": E6_MIN_CLUSTER_SIZE,
        "load_balance_coef": E6_OUTER_LOAD_BALANCE_COEF,
        "cluster_merge_map": {
            str(source): int(target)
            for source, target in sorted(merge_map.items())
        },
    }
    write_json(run_dir / "config.json", config)
    manifest = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "experiment_id": "E6",
        "snapshot_id": context.snapshot.name,
        "selected_factor_count": len(context.selected_ids),
        "cluster_count": cluster_count,
        "representative_factor_count": len(representatives),
        "inner_router_count": len(inner_models),
        "selected_factor_ids": context.selected_ids,
        "representatives": representatives,
        "test_start": str(context.test_start),
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


def run_experiments(
    cluster_run: str | Path,
    output_root: str | Path,
    experiments: str | Iterable[str] | None = None,
    top_n: int = 20,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.001,
    rolling_window: int = 40,
    epochs: int = 50,
    random_state: int = 42,
    factor_top_k: int | None = None,
) -> dict[str, Any]:
    """运行指定实验，E1–E4 共用一个套件目录，E5、E6 各自独立。"""
    selected = parse_experiments(experiments)
    cluster_dir = Path(cluster_run).expanduser().resolve()
    cluster_manifest, clustering, factor_groups, context = _load_cluster_artifacts(
        cluster_dir
    )
    representatives = [str(item) for item in cluster_manifest["representatives"]]
    output_base = Path(output_root).expanduser().resolve()
    output_base.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "cluster_run": str(cluster_dir),
        "snapshot_id": cluster_manifest["snapshot_id"],
        "experiments": [],
    }
    suite_ids = [item for item in selected if item in {"E1", "E2", "E3", "E4"}]
    if suite_ids:
        suite = _run_e1_e4(
            context,
            clustering,
            representatives,
            suite_ids,
            output_base,
            cluster_dir,
            top_n,
            fee_rate,
            slippage_rate,
            rolling_window,
            epochs,
            random_state,
            factor_top_k,
        )
        results["e1_e4"] = suite
        results["experiments"].extend(suite["experiments"])
    if "E5" in selected:
        e5 = _run_e5(
            context,
            clustering,
            factor_groups,
            representatives,
            output_base,
            cluster_dir,
            top_n,
            fee_rate,
            slippage_rate,
            rolling_window,
            epochs,
            random_state,
        )
        results["e5"] = e5
        results["experiments"].append(
            {
                "experiment_id": "E5",
                "metrics": e5["metrics"],
                "directory": e5["run_dir"],
            }
        )
    if "E6" in selected:
        e6 = _run_e6(
            context,
            clustering,
            factor_groups,
            representatives,
            output_base,
            cluster_dir,
            top_n,
            fee_rate,
            slippage_rate,
            rolling_window,
            epochs,
            random_state,
        )
        results["e6"] = e6
        results["experiments"].append(
            {
                "experiment_id": "E6",
                "metrics": e6["metrics"],
                "directory": e6["run_dir"],
            }
        )
    return results
