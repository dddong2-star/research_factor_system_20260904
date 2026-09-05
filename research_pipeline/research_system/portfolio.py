"""Simple long-only next-day execution backtester with explicit costs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _require(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def _is_suspended(frame: pd.DataFrame) -> pd.Series:
    for name in ("Ifsuspend", "if_suspend", "suspended"):
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce").fillna(0).ne(0)
    return pd.Series(False, index=frame.index)


def _desired_weights(signal: pd.DataFrame, top_n: int) -> dict[str, float]:
    clean = signal[["code", "alpha"]].copy()
    clean["alpha"] = pd.to_numeric(clean["alpha"], errors="coerce")
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna().sort_values(
        ["alpha", "code"], ascending=[False, True]
    )
    clean = clean.drop_duplicates("code").head(top_n)
    if clean.empty:
        return {}
    weight = 1.0 / len(clean)
    return {str(code): weight for code in clean["code"]}


def _apply_freeze(
    desired: dict[str, float],
    previous: dict[str, float],
    current: pd.DataFrame,
) -> dict[str, float]:
    available = set(current.loc[~_is_suspended(current), "code"].astype(str))
    frozen = {
        code: float(weight)
        for code, weight in previous.items()
        if code not in available and weight != 0
    }
    remaining = max(0.0, 1.0 - sum(frozen.values()))
    available_desired = {code: weight for code, weight in desired.items() if code in available}
    desired_total = sum(available_desired.values())
    result = dict(frozen)
    if desired_total > 0:
        result.update({code: weight * remaining / desired_total for code, weight in available_desired.items()})
    return {code: weight for code, weight in result.items() if abs(weight) > 1e-12}


def _metrics(daily: pd.DataFrame, fee_rate: float, slippage_rate: float) -> dict[str, float | int]:
    if daily.empty:
        return {
            "total_return": 0.0, "annual_return": 0.0, "sharpe": float("nan"),
            "max_drawdown": 0.0, "calmar": float("nan"), "win_rate": float("nan"),
            "average_turnover": 0.0, "total_cost": 0.0, "annual_cost": 0.0,
            "number_of_trades": 0, "average_holding_count": 0.0,
        }
    net = pd.to_numeric(daily["net_return"], errors="coerce").fillna(0.0)
    equity = (1.0 + net).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    periods = len(net)
    total = float(equity.iloc[-1] - 1.0)
    annual = float(equity.iloc[-1] ** (252.0 / periods) - 1.0) if equity.iloc[-1] > 0 else -1.0
    std = float(net.std(ddof=1)) if periods >= 2 else float("nan")
    sharpe = float(np.sqrt(252.0) * net.mean() / std) if np.isfinite(std) and std > 0 else float("nan")
    max_dd = float(drawdown.min())
    calmar = float(annual / abs(max_dd)) if max_dd < 0 else float("nan")
    return {
        "total_return": total,
        "annual_return": annual,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "win_rate": float((net > 0).mean()),
        "average_turnover": float(daily["turnover"].mean()),
        "total_cost": float(daily["cost"].sum()),
        "annual_cost": float(daily["cost"].mean() * 252.0),
        "number_of_trades": int((daily["turnover"] > 0).sum()),
        "average_holding_count": float(daily["holding_count"].mean()),
    }


def backtest_long_only(
    market: pd.DataFrame,
    alpha: pd.DataFrame,
    top_n: int,
    fee_rate: float,
    slippage_rate: float,
) -> dict[str, Any]:
    """Run a long-only backtest where each signal is executed next day."""
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if fee_rate < 0 or slippage_rate < 0:
        raise ValueError("fee_rate and slippage_rate cannot be negative")
    _require(market, {"date", "code", "close"}, "market")
    if "alpha" not in alpha.columns and "value" in alpha.columns:
        alpha = alpha.rename(columns={"value": "alpha"})
    _require(alpha, {"date", "code", "alpha"}, "alpha")
    prices = market.copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices["code"] = prices["code"].astype(str)
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    prices = prices.dropna(subset=["date"]).sort_values(["date", "code"])
    dates = pd.DatetimeIndex(prices["date"].drop_duplicates().sort_values())
    next_date = {dates[i]: dates[i + 1] for i in range(len(dates) - 1)}
    signals = alpha.copy()
    signals["date"] = pd.to_datetime(signals["date"], errors="coerce")
    signals["code"] = signals["code"].astype(str)
    execution_signals: dict[pd.Timestamp, tuple[pd.Timestamp, pd.DataFrame]] = {}
    for signal_date, group in signals.dropna(subset=["date"]).groupby("date", sort=True):
        if signal_date in next_date:
            execution_signals[next_date[signal_date]] = (signal_date, group)
    previous_weights: dict[str, float] = {}
    previous_close: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for execution_date in sorted(execution_signals):
        current = prices[prices["date"] == execution_date].copy()
        signal_date, signal_frame = execution_signals[execution_date]
        desired = _desired_weights(signal_frame, top_n)
        actual = _apply_freeze(desired, previous_weights, current)
        close_by_code = current.set_index("code")["close"].to_dict()
        if not previous_close:
            reference = prices[prices["date"] == signal_date]
            previous_close = reference.set_index("code")["close"].to_dict()
        returns: dict[str, float] = {}
        for code, close in close_by_code.items():
            old = previous_close.get(code)
            returns[code] = float(close / old - 1.0) if old and np.isfinite(close) else 0.0
        gross = float(sum(actual.get(code, 0.0) * returns.get(code, 0.0) for code in actual))
        turnover = float(sum(abs(actual.get(code, 0.0) - previous_weights.get(code, 0.0)) for code in set(actual) | set(previous_weights)))
        cost = turnover * (fee_rate + slippage_rate)
        rows.append({
            "date": execution_date,
            "signal_date": signal_date,
            "gross_return": gross,
            "turnover": turnover,
            "cost": cost,
            "net_return": gross - cost,
            "holding_count": int(sum(weight > 0 for weight in actual.values())),
            "weights": actual,
        })
        previous_weights = actual
        previous_close = {code: float(value) for code, value in close_by_code.items() if pd.notna(value)}
    daily = pd.DataFrame(rows)
    if not daily.empty:
        daily["equity"] = (1.0 + daily["net_return"]).cumprod()
    return {"daily": daily, "metrics": _metrics(daily, fee_rate, slippage_rate)}
