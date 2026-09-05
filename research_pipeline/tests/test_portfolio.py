import pandas as pd
import pytest

from research_pipeline.research_system.portfolio import backtest_long_only


def _market(suspended=False):
    dates = pd.date_range("2024-01-01", periods=3)
    rows = []
    prices = {"A": [100.0, 110.0, 110.0], "B": [100.0, 100.0, 100.0]}
    for date in dates:
        for code, closes in prices.items():
            rows.append({
                "date": date,
                "code": code,
                "open": closes[(date - dates[0]).days],
                "close": closes[(date - dates[0]).days],
                "Ifsuspend": 1 if suspended and code == "A" and date == dates[2] else 0,
            })
    return pd.DataFrame(rows)


def test_backtest_executes_next_day_and_charges_turnover_cost():
    alpha = pd.DataFrame(
        {"date": [pd.Timestamp("2024-01-01")], "code": ["A"], "alpha": [1.0]}
    )
    result = backtest_long_only(
        _market(), alpha, top_n=1, fee_rate=0.005, slippage_rate=0.005
    )

    first = result["daily"].iloc[0]
    assert first["date"] == pd.Timestamp("2024-01-02")
    assert first["gross_return"] == pytest.approx(0.10)
    assert first["cost"] == pytest.approx(0.01)
    assert first["net_return"] == pytest.approx(0.09)


def test_backtest_freezes_suspended_holding():
    dates = pd.date_range("2024-01-01", periods=3)
    alpha = pd.DataFrame(
        {
            "date": dates[:2].tolist(),
            "code": ["A", "B"],
            "alpha": [1.0, 1.0],
        }
    )
    result = backtest_long_only(
        _market(suspended=True), alpha, top_n=1, fee_rate=0.0, slippage_rate=0.0
    )
    second = result["daily"].iloc[1]
    assert second["weights"].get("A", 0.0) == pytest.approx(1.0)
