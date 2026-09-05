"""Experiment definitions for the staged dynamic factor study."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from .behavior import build_daily_factor_performance, build_rolling_performance_features
from .clustering import fit_behavior_kmeans, select_representatives
from .features import make_forward_return
from .io import resolve_snapshot, write_json
from .models import FactorRouter, gumbel_topk_mask, turnover_penalty
from .portfolio import backtest_long_only


def build_experiment_specs(
    factor_ids: Iterable[str],
    test_start: object,
) -> list[dict[str, Any]]:
    ids = [str(item) for item in factor_ids]
    common = {"factor_ids": ids, "test_start": str(test_start)}
    return [
        {
            **common,
            "experiment_id": "E1",
            "name": "K-Means equal-weight baseline",
            "components": ["kmeans", "equal_weight", "cost_backtest"],
        },
        {
            **common,
            "experiment_id": "E2",
            "name": "Market-aware routing",
            "components": ["kmeans", "macoe", "cost_backtest"],
        },
        {
            **common,
            "experiment_id": "E3",
            "name": "Performance-aware routing",
            "components": ["kmeans", "macoe", "pacoe", "cost_backtest"],
        },
        {
            **common,
            "experiment_id": "E4",
            "name": "Full fused router",
            "components": [
                "kmeans", "macoe", "pacoe", "gumbel_topk", "turnover_penalty",
                "cost_backtest",
            ],
        },
    ]


def _check_snapshot(run_dir: Path, manifest_name: str, snapshot_id: str) -> None:
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
    for source, target in (("ic", "ic"), ("rank_ic", "rank_ic"), ("slope", "slope")):
        values = pd.to_numeric(group[source], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        mean = float(values.mean()) if not values.empty else np.nan
        std = float(values.std(ddof=1)) if len(values) >= 2 else np.nan
        row[f"{target}_mean"] = mean
        row[f"{target}_std"] = std
        row[f"{target}_ir"] = float(mean / std) if np.isfinite(std) and std > 0 else np.nan
    return row


def _behavior_frame(
    performance: pd.DataFrame,
    factor_ids: list[str],
    train_end: pd.Timestamp,
) -> pd.DataFrame:
    mature = performance[
        performance["date"].lt(train_end)
        & performance["label_available_date"].lt(train_end)
    ]
    grouped = {str(key): value for key, value in mature.groupby("factor_id")}
    return pd.DataFrame(
        [_summary_row(grouped[factor_id], factor_id) if factor_id in grouped else {"factor_id": factor_id} for factor_id in factor_ids]
    ).set_index("factor_id")


def _rolling_table(
    performance: pd.DataFrame,
    factor_ids: list[str],
    dates: Iterable[pd.Timestamp],
    window: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for date in pd.DatetimeIndex(dates).sort_values():
        features = build_rolling_performance_features(performance, date, window=window)
        if not features.empty:
            features = features[features["factor_id"].isin(factor_ids)].copy()
            features["date"] = date
            rows.append(features)
    if not rows:
        return pd.DataFrame(columns=["date", "factor_id"])
    return pd.concat(rows, ignore_index=True)


def _market_feature_frame(states: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    columns = ["market_return", "market_volatility", "up_ratio", "cross_section_dispersion"]
    frame = states.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in columns:
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.set_index("date")[columns].reindex(dates)
    return frame.fillna(0.0)


def _performance_tensor(
    rolling: pd.DataFrame,
    dates: pd.DatetimeIndex,
    factor_ids: list[str],
) -> torch.Tensor:
    columns = [
        "ic_mean", "ic_std", "ic_ir", "rank_ic_mean", "rank_ic_std",
        "rank_ic_ir", "slope_mean", "slope_std", "slope_ir",
    ]
    tensor = np.zeros((len(dates), len(factor_ids), len(columns)), dtype=np.float32)
    if rolling.empty:
        return torch.from_numpy(tensor)
    indexed = rolling.set_index(["date", "factor_id"])
    for date_index, date in enumerate(dates):
        for factor_index, factor_id in enumerate(factor_ids):
            key = (date, factor_id)
            if key in indexed.index:
                values = pd.to_numeric(indexed.loc[key, columns], errors="coerce").fillna(0.0)
                tensor[date_index, factor_index] = values.to_numpy(dtype=np.float32)
    return torch.from_numpy(tensor)


def _target_tensor(
    performance: pd.DataFrame,
    dates: pd.DatetimeIndex,
    factor_ids: list[str],
) -> torch.Tensor:
    pivot = performance.pivot_table(index="date", columns="factor_id", values="rank_ic", aggfunc="mean")
    pivot = pivot.reindex(index=dates, columns=factor_ids)
    return torch.tensor(pivot.to_numpy(dtype=np.float32), dtype=torch.float32)


def _fit_router(
    market_features: pd.DataFrame,
    performance_input: torch.Tensor,
    target: torch.Tensor,
    train_mask: np.ndarray,
    factor_count: int,
    epochs: int,
    random_state: int,
    use_performance: bool,
    use_turnover_penalty: bool,
    factor_top_k: int,
) -> FactorRouter:
    torch.manual_seed(random_state)
    model = FactorRouter(4, 9, factor_count, hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    market_tensor = torch.tensor(market_features.to_numpy(dtype=np.float32))
    valid = torch.isfinite(target) & torch.tensor(train_mask[:, None])
    train_target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    route_input = performance_input if use_performance else torch.zeros_like(performance_input)
    previous = torch.zeros((factor_count,), dtype=torch.float32)
    for _ in range(max(1, epochs)):
        optimizer.zero_grad()
        output = model(market_tensor, route_input)
        scores = output["fused_scores"] if use_performance else output["market_scores"]
        routed_scores = scores
        if use_turnover_penalty:
            mask = gumbel_topk_mask(
                scores, min(factor_top_k, factor_count), temperature=0.5, training=True
            )
            routed_scores = scores * mask
        loss_matrix = (routed_scores - train_target).pow(2)
        loss = loss_matrix[valid].mean() if valid.any() else loss_matrix.mean()
        if use_turnover_penalty and len(scores) > 1:
            loss = loss + 0.01 * torch.stack(
                [turnover_penalty(torch.softmax(routed_scores[i : i + 1], -1), previous.unsqueeze(0)) for i in range(len(scores))]
            ).mean()
        loss.backward()
        optimizer.step()
        previous = torch.softmax(routed_scores.detach()[-1], -1)
    return model


def _route_weights(
    model: FactorRouter | None,
    market_features: pd.DataFrame,
    performance_input: torch.Tensor,
    dates: pd.DatetimeIndex,
    factor_ids: list[str],
    route: str,
    top_k: int,
) -> pd.DataFrame:
    if model is None:
        values = np.full((len(dates), len(factor_ids)), 1.0 / len(factor_ids))
    else:
        model.eval()
        with torch.no_grad():
            output = model(
                torch.tensor(market_features.to_numpy(dtype=np.float32)), performance_input
            )
            if route == "market":
                scores = output["market_scores"]
            else:
                scores = output["fused_scores"]
            if route == "full":
                mask = gumbel_topk_mask(scores, min(top_k, len(factor_ids)), 0.5, training=False)
                scores = scores.masked_fill(mask.eq(0), -1e9)
            values = torch.softmax(scores, dim=-1).cpu().numpy()
    return pd.DataFrame(
        [
            {"date": date, "factor_id": factor_id, "weight": float(values[i, j])}
            for i, date in enumerate(dates)
            for j, factor_id in enumerate(factor_ids)
        ]
    )


def _build_alpha(values: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    wide = values.pivot_table(index=["date", "code"], columns="factor_id", values="value", aggfunc="mean")
    wide = wide.sort_index()
    weight_wide = weights.pivot_table(index="date", columns="factor_id", values="weight", aggfunc="mean")
    result: list[pd.DataFrame] = []
    for date, group in wide.groupby(level="date", sort=True):
        if date not in weight_wide.index:
            continue
        factor_values = group.droplevel("date")
        available_weights = weight_wide.reindex(columns=factor_values.columns).loc[date].fillna(0.0)
        ranks = factor_values.replace([np.inf, -np.inf], np.nan)
        means = ranks.mean(axis=0)
        scales = ranks.std(axis=0, ddof=0).replace(0, np.nan)
        standardized = (ranks - means) / scales
        numerator = standardized.mul(available_weights, axis=1).sum(axis=1, min_count=1)
        denominator = standardized.notna().mul(available_weights, axis=1).sum(axis=1)
        alpha = numerator.div(denominator.replace(0, np.nan)).rename("alpha").reset_index()
        alpha["date"] = date
        result.append(alpha[["date", "code", "alpha"]])
    return pd.concat(result, ignore_index=True) if result else pd.DataFrame(columns=["date", "code", "alpha"])


def run_experiment_suite(
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
    factor_top_k: int | None = None,
) -> dict[str, Any]:
    """Run E1-E4 on one immutable snapshot and save auditable artifacts."""
    snapshot = resolve_snapshot(data_root, snapshot_id)
    values_root = Path(values_run).expanduser().resolve() / "factor_values"
    state_path = Path(state_run).expanduser().resolve() / "market_states.parquet"
    _check_snapshot(values_root.parent, "factor_values_manifest.json", snapshot.name)
    _check_snapshot(state_path.parent, "manifest.json", snapshot.name)
    if not values_root.exists() or not state_path.exists():
        raise FileNotFoundError("values_run or state_run does not contain required files")
    market = pd.read_parquet(snapshot / "market.parquet")
    states = pd.read_parquet(state_path)
    partitions = sorted(values_root.glob("factor_id=*/values.parquet"))
    available = [path.parent.name.split("=", 1)[-1] for path in partitions]
    requested = [str(item) for item in factor_ids] if factor_ids is not None else available
    selected_ids = [factor_id for factor_id in available if factor_id in set(requested)]
    if max_factors is not None:
        if max_factors <= 0:
            raise ValueError("max_factors must be positive")
        selected_ids = selected_ids[:max_factors]
    if not selected_ids:
        raise ValueError("no factor partitions selected")
    if factor_top_k is None:
        factor_top_k = len(selected_ids)
    if factor_top_k <= 0:
        raise ValueError("factor_top_k must be positive")
    target = make_forward_return(market, horizon=horizon)
    performances: list[pd.DataFrame] = []
    values_parts: list[pd.DataFrame] = []
    for path, factor_id in zip(partitions, available):
        if factor_id not in selected_ids:
            continue
        values = pd.read_parquet(path, columns=["date", "code", "value"])
        values["factor_id"] = factor_id
        values_parts.append(values)
        performances.append(build_daily_factor_performance(values, target, horizon=horizon))
    performance = pd.concat(performances, ignore_index=True)
    all_dates = pd.DatetimeIndex(pd.to_datetime(market["date"], errors="coerce").dropna().unique()).sort_values()
    if test_start is None:
        test_start_ts = all_dates[max(1, int(len(all_dates) * 0.8))]
    else:
        test_start_ts = pd.Timestamp(test_start)
    train_end = test_start_ts
    behavior = _behavior_frame(performance, selected_ids, train_end)
    clustering = fit_behavior_kmeans(
        behavior, n_clusters=min(n_clusters, len(selected_ids)), random_state=random_state
    )
    representatives = select_representatives(
        clustering["labels"], behavior, clustering["centers"], per_cluster=1
    )
    representatives = [factor_id for factor_id in representatives if factor_id in selected_ids]
    if not representatives:
        representatives = selected_ids[:1]
    candidate_values = pd.concat(
        [frame[frame["factor_id"].isin(representatives)] for frame in values_parts],
        ignore_index=True,
    )
    signal_dates = all_dates[all_dates >= test_start_ts]
    rolling = _rolling_table(performance, representatives, all_dates, rolling_window)
    market_features = _market_feature_frame(states, all_dates)
    perf_tensor = _performance_tensor(rolling, all_dates, representatives)
    target_tensor = _target_tensor(performance, all_dates, representatives)
    train_mask = all_dates < train_end
    output_base = Path(output_root).expanduser().resolve()
    seed = json.dumps({"snapshot": snapshot.name, "factors": representatives, "test_start": str(test_start_ts), "random_state": random_state}, sort_keys=True)
    run_id = f"experiments_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{hashlib.sha256(seed.encode()).hexdigest()[:8]}"
    run_dir = output_base / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    performance.to_parquet(run_dir / "daily_factor_performance.parquet", index=False)
    behavior.reset_index().to_parquet(run_dir / "behavior_features.parquet", index=False)
    pd.DataFrame({"factor_id": selected_ids, "cluster": clustering["labels"]}).to_parquet(
        run_dir / "factor_clusters.parquet", index=False
    )
    write_json(
        run_dir / "clustering.json",
        {
            "n_clusters": int(min(n_clusters, len(selected_ids))),
            "feature_columns": clustering["feature_columns"],
            "centers": clustering["centers"].tolist(),
            "inertia": clustering["inertia"],
            "random_state": random_state,
            "representatives": representatives,
            "training_end": str(train_end),
        },
    )
    specs = build_experiment_specs(representatives, test_start_ts)
    outputs: list[dict[str, Any]] = []
    for spec in specs:
        experiment_id = spec["experiment_id"]
        experiment_dir = run_dir / experiment_id
        experiment_dir.mkdir()
        if experiment_id == "E1":
            weights = pd.DataFrame(
                [{"date": date, "factor_id": factor_id, "weight": 1.0 / len(representatives)} for date in all_dates for factor_id in representatives]
            )
            model = None
            route = "equal"
        else:
            use_performance = experiment_id in {"E3", "E4"}
            model = _fit_router(
                market_features, perf_tensor, target_tensor, train_mask,
                len(representatives), epochs, random_state,
                use_performance=use_performance,
                use_turnover_penalty=experiment_id == "E4",
                factor_top_k=min(factor_top_k, len(representatives)),
            )
            route = "market" if experiment_id == "E2" else ("full" if experiment_id == "E4" else "fused")
            weights = _route_weights(
                model, market_features, perf_tensor, all_dates, representatives,
                route, top_k=min(factor_top_k, len(representatives)),
            )
            torch.save(model.state_dict(), experiment_dir / "model.pt")
        weights.to_parquet(experiment_dir / "factor_weights.parquet", index=False)
        test_weights = weights[weights["date"].ge(test_start_ts)]
        alpha = _build_alpha(candidate_values, test_weights)
        alpha.to_parquet(experiment_dir / "alpha.parquet", index=False)
        backtest = backtest_long_only(
            market, alpha, top_n=top_n, fee_rate=fee_rate, slippage_rate=slippage_rate
        )
        backtest["daily"].to_parquet(experiment_dir / "daily.parquet", index=False)
        write_json(experiment_dir / "metrics.json", backtest["metrics"])
        config = {**spec, "snapshot_id": snapshot.name, "horizon": horizon, "n_clusters": n_clusters, "representatives": representatives, "fee_rate": fee_rate, "slippage_rate": slippage_rate, "rolling_window": rolling_window, "epochs": epochs, "factor_top_k": factor_top_k, "random_state": random_state}
        write_json(experiment_dir / "config.json", config)
        outputs.append({"experiment_id": experiment_id, "components": spec["components"], "metrics": backtest["metrics"], "directory": str(experiment_dir)})
    manifest = {
        "run_id": run_id,
        "snapshot_id": snapshot.name,
        "selected_factor_count": len(selected_ids),
        "representative_factor_count": len(representatives),
        "selected_factor_ids": selected_ids,
        "representatives": representatives,
        "test_start": str(test_start_ts),
        "training_end": str(train_end),
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
