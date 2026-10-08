"""E1–E4 实验规格。"""

from __future__ import annotations

from typing import Any, Iterable


def build_experiment_specs(
    factor_ids: Iterable[str],
    test_start: object,
) -> list[dict[str, Any]]:
    """返回保持原顺序和组件名称的 E1–E4 规格。"""
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
                "kmeans",
                "macoe",
                "pacoe",
                "gumbel_topk",
                "turnover_penalty",
                "cost_backtest",
            ],
        },
    ]
