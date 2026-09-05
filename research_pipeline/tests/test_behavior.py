import pandas as pd
import pytest

from research_pipeline.research_system.behavior import (
    build_daily_factor_performance,
    build_rolling_performance_features,
)


def test_build_daily_factor_performance_returns_ic_rank_ic_slope_and_mature_date():
    dates = pd.date_range("2024-01-01", periods=5)
    values = pd.DataFrame([
        {"date": d, "code": code, "factor_id": "f1", "value": float(v)}
        for d in dates
        for code, v in [("000001", 1), ("000002", 2), ("000003", 3)]
    ])
    target = pd.DataFrame([
        {"date": d, "code": code, "target": float(v)}
        for d in dates
        for code, v in [("000001", 0.01), ("000002", 0.02), ("000003", 0.03)]
    ])

    result = build_daily_factor_performance(values, target, horizon=20)

    assert {"date", "factor_id", "ic", "rank_ic", "slope", "sample_count", "label_available_date"}.issubset(result.columns)
    assert result.iloc[0]["ic"] == pytest.approx(1.0)
    assert result.iloc[0]["rank_ic"] == pytest.approx(1.0)
    assert result.iloc[0]["label_available_date"] == dates[0] + pd.Timedelta(days=20)


def test_rolling_performance_features_only_uses_mature_labels():
    dates = pd.date_range("2024-01-01", periods=4)
    performance = pd.DataFrame({
        "date": dates,
        "factor_id": ["f1"] * 4,
        "ic": [1.0, 0.5, -0.5, 0.25],
        "rank_ic": [0.8, 0.4, -0.4, 0.2],
        "slope": [2.0, 1.0, -1.0, 0.5],
        "sample_count": [3] * 4,
        "label_available_date": dates + pd.Timedelta(days=2),
    })

    result = build_rolling_performance_features(performance, as_of_date=dates[3], window=10)

    assert result.iloc[0]["ic_mean"] == pytest.approx(1.0)
    assert result.iloc[0]["rank_ic_mean"] == pytest.approx(0.8)
    assert result.iloc[0]["slope_mean"] == pytest.approx(2.0)
