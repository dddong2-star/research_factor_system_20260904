"""校验研究快照的完整性与清单哈希。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..common.io import file_sha256, resolve_snapshot, validate_unique_keys
from ..common.schema import (
    BACKTEST_METRIC_REQUIRED_COLUMNS,
    FACTOR_METRIC_REQUIRED_COLUMNS,
    FACTOR_REQUIRED_COLUMNS,
    MARKET_REQUIRED_COLUMNS,
    TABLE_FILENAMES,
    UNIVERSE_REQUIRED_COLUMNS,
)


def validate_snapshot(
    data_root: str | Path,
    snapshot_id: str | None = None,
) -> dict[str, Any]:
    """校验快照文件、必需列、主键和清单哈希。"""
    errors: list[str] = []
    try:
        snapshot = resolve_snapshot(data_root, snapshot_id)
    except (FileNotFoundError, ValueError) as exc:
        return {"valid": False, "errors": [str(exc)], "snapshot": None}
    manifest_path = snapshot / "manifest.json"
    if not manifest_path.exists():
        return {
            "valid": False,
            "errors": [f"manifest is missing: {manifest_path}"],
            "snapshot": str(snapshot),
        }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "valid": False,
            "errors": [f"invalid manifest JSON: {exc}"],
            "snapshot": str(snapshot),
        }
    if manifest.get("snapshot_id") != snapshot.name:
        errors.append("manifest snapshot_id does not match directory name")
    listed = {
        str(item.get("path")): item
        for item in manifest.get("files", [])
        if isinstance(item, dict)
    }
    for filename in TABLE_FILENAMES:
        path = snapshot / filename
        if not path.exists():
            errors.append(f"required table is missing: {filename}")
            continue
        item = listed.get(filename)
        if item is None:
            errors.append(f"table is missing from manifest: {filename}")
        elif item.get("sha256") != file_sha256(path):
            errors.append(f"SHA-256 mismatch: {filename}")
    errors.extend(_validate_tables(snapshot))
    return {
        "valid": not errors,
        "errors": errors,
        "snapshot": str(snapshot),
        "snapshot_id": snapshot.name,
        "manifest_valid_flag": bool(manifest.get("valid", False)),
        "n_codes": manifest.get("n_codes"),
        "factor_count": manifest.get("factor_count"),
    }


def _validate_tables(snapshot: Path) -> list[str]:
    errors: list[str] = []
    try:
        market = pd.read_parquet(snapshot / "market.parquet")
        missing = set(MARKET_REQUIRED_COLUMNS) - set(market.columns)
        if missing:
            errors.append(f"market.parquet is missing columns: {sorted(missing)}")
        else:
            validate_unique_keys(market, ("date", "code"))
    except Exception as exc:
        errors.append(f"market.parquet validation failed: {exc}")
    try:
        universe = pd.read_csv(snapshot / "universe.csv")
        missing = set(UNIVERSE_REQUIRED_COLUMNS) - set(universe.columns)
        if missing:
            errors.append(f"universe.csv is missing columns: {sorted(missing)}")
        elif universe["code"].duplicated().any():
            errors.append("universe.csv contains duplicate codes")
    except Exception as exc:
        errors.append(f"universe.csv validation failed: {exc}")
    try:
        catalog = pd.read_parquet(snapshot / "factor_catalog.parquet")
        missing = set(FACTOR_REQUIRED_COLUMNS) - set(catalog.columns)
        if missing:
            errors.append(f"factor_catalog.parquet is missing columns: {sorted(missing)}")
    except Exception as exc:
        errors.append(f"factor_catalog.parquet validation failed: {exc}")
    try:
        metrics = pd.read_parquet(snapshot / "factor_metrics.parquet")
        missing = set(FACTOR_METRIC_REQUIRED_COLUMNS) - set(metrics.columns)
        if missing:
            errors.append(f"factor_metrics.parquet is missing columns: {sorted(missing)}")
    except Exception as exc:
        errors.append(f"factor_metrics.parquet validation failed: {exc}")
    try:
        backtests = pd.read_parquet(snapshot / "backtest_metrics.parquet")
        missing = set(BACKTEST_METRIC_REQUIRED_COLUMNS) - set(backtests.columns)
        if missing:
            errors.append(
                f"backtest_metrics.parquet is missing columns: {sorted(missing)}"
            )
    except Exception as exc:
        errors.append(f"backtest_metrics.parquet validation failed: {exc}")
    return errors
