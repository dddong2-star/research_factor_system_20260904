"""将因子值与动态权重合成为股票 Alpha。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_alpha(
    values: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.DataFrame:
    """逐日标准化因子值，并按可用权重合成 Alpha。"""
    wide = values.pivot_table(
        index=["date", "code"],
        columns="factor_id",
        values="value",
        aggfunc="mean",
    ).sort_index()
    weight_wide = weights.pivot_table(
        index="date",
        columns="factor_id",
        values="weight",
        aggfunc="mean",
    )
    result: list[pd.DataFrame] = []
    for date, group in wide.groupby(level="date", sort=True):
        if date not in weight_wide.index:
            continue
        factor_values = group.droplevel("date")
        available_weights = (
            weight_wide.reindex(columns=factor_values.columns)
            .loc[date]
            .fillna(0.0)
        )
        ranks = factor_values.replace([np.inf, -np.inf], np.nan)
        means = ranks.mean(axis=0)
        scales = ranks.std(axis=0, ddof=0).replace(0, np.nan)
        standardized = (ranks - means) / scales
        numerator = standardized.mul(
            available_weights,
            axis=1,
        ).sum(axis=1, min_count=1)
        denominator = standardized.notna().mul(
            available_weights,
            axis=1,
        ).sum(axis=1)
        alpha = (
            numerator.div(denominator.replace(0, np.nan))
            .rename("alpha")
            .reset_index()
        )
        alpha["date"] = date
        result.append(alpha[["date", "code", "alpha"]])
    if not result:
        return pd.DataFrame(columns=["date", "code", "alpha"])
    return pd.concat(result, ignore_index=True)
