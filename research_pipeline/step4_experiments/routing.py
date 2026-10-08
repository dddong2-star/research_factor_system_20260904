"""E1–E6 共用的路由特征构造、训练和推理。"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import torch

from ..common.behavior import build_rolling_performance_features
from ..common.models import FactorRouter, gumbel_topk_mask, turnover_penalty


def rolling_table(
    performance: pd.DataFrame,
    factor_ids: list[str],
    dates: Iterable[pd.Timestamp],
    window: int,
) -> pd.DataFrame:
    """按日期生成指定因子的滚动绩效特征表。"""
    rows: list[pd.DataFrame] = []
    for date in pd.DatetimeIndex(dates).sort_values():
        features = build_rolling_performance_features(
            performance,
            date,
            window=window,
        )
        if not features.empty:
            features = features[
                features["factor_id"].isin(factor_ids)
            ].copy()
            features["date"] = date
            rows.append(features)
    if not rows:
        return pd.DataFrame(columns=["date", "factor_id"])
    return pd.concat(rows, ignore_index=True)


def market_feature_frame(
    states: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """按统一日期轴提取四维市场状态特征。"""
    columns = [
        "market_return",
        "market_volatility",
        "up_ratio",
        "cross_section_dispersion",
    ]
    frame = states.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in columns:
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.set_index("date")[columns].reindex(dates).fillna(0.0)


def performance_tensor(
    rolling: pd.DataFrame,
    dates: pd.DatetimeIndex,
    factor_ids: list[str],
) -> torch.Tensor:
    """将九维滚动绩效特征转换为 `[日期, 因子, 特征]` 张量。"""
    columns = [
        "ic_mean",
        "ic_std",
        "ic_ir",
        "rank_ic_mean",
        "rank_ic_std",
        "rank_ic_ir",
        "slope_mean",
        "slope_std",
        "slope_ir",
    ]
    tensor = np.zeros(
        (len(dates), len(factor_ids), len(columns)),
        dtype=np.float32,
    )
    if rolling.empty:
        return torch.from_numpy(tensor)
    indexed = rolling.set_index(["date", "factor_id"])
    for date_index, date in enumerate(dates):
        for factor_index, factor_id in enumerate(factor_ids):
            key = (date, factor_id)
            if key in indexed.index:
                values = (
                    pd.to_numeric(indexed.loc[key, columns], errors="coerce")
                    .fillna(0.0)
                    .to_numpy(dtype=np.float32)
                )
                tensor[date_index, factor_index] = values
    return torch.from_numpy(tensor)


def target_tensor(
    performance: pd.DataFrame,
    dates: pd.DatetimeIndex,
    factor_ids: list[str],
) -> torch.Tensor:
    """将因子日度 RankIC 目标转换为二维张量。"""
    pivot = performance.pivot_table(
        index="date",
        columns="factor_id",
        values="rank_ic",
        aggfunc="mean",
    )
    pivot = pivot.reindex(index=dates, columns=factor_ids)
    return torch.tensor(
        pivot.to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )


def fit_router(
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
    load_balance_coef: float = 0.0,
) -> FactorRouter:
    """按原 E2–E4 训练逻辑拟合一个因子路由器。"""
    torch.manual_seed(random_state)
    model = FactorRouter(4, 9, factor_count, hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    market_tensor = torch.tensor(
        market_features.to_numpy(dtype=np.float32)
    )
    valid = torch.isfinite(target) & torch.tensor(train_mask[:, None])
    train_target = torch.nan_to_num(
        target,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    route_input = (
        performance_input
        if use_performance
        else torch.zeros_like(performance_input)
    )
    previous = torch.zeros((factor_count,), dtype=torch.float32)
    for _ in range(max(1, epochs)):
        optimizer.zero_grad()
        output = model(market_tensor, route_input)
        scores = (
            output["fused_scores"]
            if use_performance
            else output["market_scores"]
        )
        routed_scores = scores
        if use_turnover_penalty:
            mask = gumbel_topk_mask(
                scores,
                min(factor_top_k, factor_count),
                temperature=0.5,
                training=True,
            )
            routed_scores = scores * mask
        loss_matrix = (routed_scores - train_target).pow(2)
        loss = (
            loss_matrix[valid].mean()
            if valid.any()
            else loss_matrix.mean()
        )
        if load_balance_coef > 0 and scores.shape[-1] > 0:
            train_rows = torch.as_tensor(train_mask, dtype=torch.bool)
            if train_rows.any():
                probabilities = torch.softmax(scores[train_rows], dim=-1)
                average_probability = probabilities.mean(dim=0)
                uniform = 1.0 / float(scores.shape[-1])
                loss = loss + load_balance_coef * (
                    average_probability - uniform
                ).pow(2).mean()
        if use_turnover_penalty and len(scores) > 1:
            loss = loss + 0.01 * torch.stack(
                [
                    turnover_penalty(
                        torch.softmax(routed_scores[i : i + 1], -1),
                        previous.unsqueeze(0),
                    )
                    for i in range(len(scores))
                ]
            ).mean()
        loss.backward()
        optimizer.step()
        previous = torch.softmax(routed_scores.detach()[-1], -1)
    return model


def route_weights(
    model: FactorRouter | None,
    market_features: pd.DataFrame,
    performance_input: torch.Tensor,
    dates: pd.DatetimeIndex,
    factor_ids: list[str],
    route: str,
    top_k: int,
) -> pd.DataFrame:
    """按指定路由模式生成每日因子权重。"""
    if model is None:
        values = np.full(
            (len(dates), len(factor_ids)),
            1.0 / len(factor_ids),
        )
    else:
        model.eval()
        with torch.no_grad():
            output = model(
                torch.tensor(
                    market_features.to_numpy(dtype=np.float32)
                ),
                performance_input,
            )
            scores = (
                output["market_scores"]
                if route == "market"
                else output["fused_scores"]
            )
            if route == "full":
                mask = gumbel_topk_mask(
                    scores,
                    min(top_k, len(factor_ids)),
                    0.5,
                    training=False,
                )
                scores = scores.masked_fill(mask.eq(0), -1e9)
            values = torch.softmax(scores, dim=-1).cpu().numpy()
    return pd.DataFrame(
        [
            {
                "date": date,
                "factor_id": factor_id,
                "weight": float(values[i, j]),
            }
            for i, date in enumerate(dates)
            for j, factor_id in enumerate(factor_ids)
        ]
    )
