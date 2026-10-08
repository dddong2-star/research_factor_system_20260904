"""Daily factor behavior and strictly lagged performance features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _require(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def _corr(left: pd.Series, right: pd.Series, rank: bool = False) -> float:
    frame = pd.DataFrame({"left": left, "right": right}).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if len(frame) < 3:
        return float("nan")
    if rank:
        frame = frame.rank(method="average")
    if frame["left"].std(ddof=0) == 0 or frame["right"].std(ddof=0) == 0:
        return float("nan")
    return float(frame["left"].corr(frame["right"]))


def _slope(left: pd.Series, right: pd.Series) -> float:
    frame = pd.DataFrame({"left": left, "right": right}).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if len(frame) < 2:
        return float("nan")
    x = frame["left"].to_numpy(dtype=float)
    y = frame["right"].to_numpy(dtype=float)
    denominator = float(np.sum((x - x.mean()) ** 2))
    if denominator == 0:
        return float("nan")
    return float(np.sum((x - x.mean()) * (y - y.mean())) / denominator)


def build_daily_factor_performance(
    values: pd.DataFrame,
    target: pd.DataFrame,
    horizon: int = 20,
) -> pd.DataFrame:
    """Compute one cross-sectional performance row per factor and date.

    ``target`` is assumed to be the already constructed forward-return label.
    The maturity date is retained explicitly so downstream routing can enforce
    a strict ``label_available_date < as_of_date`` boundary.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    _require(values, {"date", "code", "factor_id", "value"}, "factor values")
    _require(target, {"date", "code", "target"}, "target")
    left = values[["date", "code", "factor_id", "value"]].copy()
    right = target[["date", "code", "target"]].copy()
    for frame in (left, right):
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["code"] = frame["code"].astype(str)
    left["value"] = pd.to_numeric(left["value"], errors="coerce")
    right["target"] = pd.to_numeric(right["target"], errors="coerce")
    merged = left.merge(right, on=["date", "code"], how="inner", validate="many_to_one")
    rows: list[dict[str, object]] = []
    for (date, factor_id), group in merged.groupby(["date", "factor_id"], sort=True):
        valid = group[["value", "target"]].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        rows.append({
            "date": pd.Timestamp(date),
            "factor_id": str(factor_id),
            "ic": _corr(valid["value"], valid["target"]),
            "rank_ic": _corr(valid["value"], valid["target"], rank=True),
            "slope": _slope(valid["value"], valid["target"]),
            "sample_count": int(len(valid)),
            "label_available_date": pd.Timestamp(date) + pd.Timedelta(days=horizon),
        })
    columns = [
        "date", "factor_id", "ic", "rank_ic", "slope", "sample_count",
        "label_available_date",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["date", "factor_id"]
    ).reset_index(drop=True)


def _summary(values: pd.Series) -> tuple[float, float, float]:
    clean = pd.to_numeric(values, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if clean.empty:
        return float("nan"), float("nan"), float("nan")
    mean = float(clean.mean())
    std = float(clean.std(ddof=1)) if len(clean) >= 2 else float("nan")
    ir = float(mean / std) if np.isfinite(std) and std > 0 else float("nan")
    return mean, std, ir


def build_rolling_performance_features(
    performance: pd.DataFrame,
    as_of_date: object,
    window: int = 40,
) -> pd.DataFrame:
    """Build 9-dimensional PACoE features using only mature past labels."""
    if window <= 0:
        raise ValueError("window must be positive")
    _require(
        performance,
        {"date", "factor_id", "ic", "rank_ic", "slope", "label_available_date"},
        "performance",
    )
    as_of = pd.Timestamp(as_of_date)
    frame = performance.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["label_available_date"] = pd.to_datetime(
        frame["label_available_date"], errors="coerce"
    )
    # Strict inequality prevents a label becoming usable on its maturity day.
    mature = frame[
        frame["date"].lt(as_of)
        & frame["label_available_date"].lt(as_of)
    ].sort_values(["factor_id", "date"])
    rows: list[dict[str, object]] = []
    for factor_id, group in mature.groupby("factor_id", sort=True):
        group = group.tail(window)
        ic_mean, ic_std, ic_ir = _summary(group["ic"])
        rank_mean, rank_std, rank_ir = _summary(group["rank_ic"])
        slope_mean, slope_std, slope_ir = _summary(group["slope"])
        rows.append({
            "as_of_date": as_of,
            "factor_id": str(factor_id),
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "ic_ir": ic_ir,
            "rank_ic_mean": rank_mean,
            "rank_ic_std": rank_std,
            "rank_ic_ir": rank_ir,
            "slope_mean": slope_mean,
            "slope_std": slope_std,
            "slope_ir": slope_ir,
            "mature_sample_count": int(len(group)),
        })
    columns = [
        "as_of_date", "factor_id", "ic_mean", "ic_std", "ic_ir",
        "rank_ic_mean", "rank_ic_std", "rank_ic_ir", "slope_mean",
        "slope_std", "slope_ir", "mature_sample_count",
    ]
    return pd.DataFrame(rows, columns=columns).reset_index(drop=True)
