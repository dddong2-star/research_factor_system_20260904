"""E5/E6 共用的因子群构建、选群和群内权重处理。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..common.io import write_json
from .clustering import select_representatives

# E6 外层选群：主簇最短停留交易日数。soft_top_k=1 表示只用主簇，关掉 Top-2。
E6_MIN_STAY_DAYS = 20
E6_SOFT_TOP_K = 1
# E5/E6 运行时合并人数少于此值的簇；外层 Softmax 均匀负载均衡系数。
E6_MIN_CLUSTER_SIZE = 3
E6_OUTER_LOAD_BALANCE_COEF = 0.01


def build_factor_group_table(
    features: pd.DataFrame,
    labels: np.ndarray | list[int],
    centers: np.ndarray,
) -> pd.DataFrame:
    """保存全部聚类成员，并将离中心最近的因子标记为代表因子。"""
    labels_array = np.asarray(labels, dtype=int)
    centers_array = np.asarray(centers, dtype=float)
    if len(features) != len(labels_array):
        raise ValueError("labels and features must have the same row count")

    numeric_columns = [
        column
        for column in features.columns
        if column not in {"factor_id", "date", "as_of_date"}
        and pd.api.types.is_numeric_dtype(features[column])
    ]
    if not numeric_columns:
        raise ValueError("features has no numeric feature columns")
    if centers_array.ndim != 2 or centers_array.shape[1] != len(
        numeric_columns
    ):
        raise ValueError("centers do not match numeric feature columns")

    matrix = features[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    matrix = matrix.replace([np.inf, -np.inf], np.nan)
    matrix = matrix.fillna(matrix.median()).fillna(0.0)
    factor_ids = (
        features["factor_id"].astype(str).tolist()
        if "factor_id" in features.columns
        else [str(item) for item in features.index]
    )
    representatives = select_representatives(
        labels_array,
        features,
        centers_array,
        per_cluster=1,
    )
    factor_to_cluster = dict(zip(factor_ids, labels_array.tolist()))
    representative_by_cluster = {
        int(factor_to_cluster[factor_id]): factor_id
        for factor_id in representatives
    }

    rows: list[dict[str, Any]] = []
    for position, factor_id in enumerate(factor_ids):
        cluster = int(labels_array[position])
        distance = float(
            np.linalg.norm(
                matrix.iloc[position].to_numpy(dtype=float)
                - centers_array[cluster]
            )
        )
        representative = representative_by_cluster[cluster]
        rows.append(
            {
                "cluster": cluster,
                "factor_id": factor_id,
                "representative_factor_id": representative,
                "is_representative": factor_id == representative,
                "distance_to_center": distance,
            }
        )

    result = pd.DataFrame(rows)
    result["group_size"] = (
        result.groupby("cluster")["factor_id"]
        .transform("size")
        .astype(int)
    )
    return result.sort_values(
        ["cluster", "distance_to_center", "factor_id"],
        kind="stable",
    ).reset_index(drop=True)


def select_top_representative_group(
    representative_weights: pd.DataFrame,
    factor_groups: pd.DataFrame,
) -> pd.DataFrame:
    """按日期选择外层权重最高的代表因子及其所属因子群。"""
    required_weights = {"date", "factor_id", "weight"}
    required_groups = {
        "cluster",
        "factor_id",
        "representative_factor_id",
        "is_representative",
        "group_size",
    }
    if missing := required_weights - set(representative_weights.columns):
        raise ValueError(
            f"representative_weights missing columns: {sorted(missing)}"
        )
    if missing := required_groups - set(factor_groups.columns):
        raise ValueError(
            f"factor_groups missing columns: {sorted(missing)}"
        )

    representative_rows = factor_groups[
        factor_groups["is_representative"]
    ][
        ["cluster", "representative_factor_id", "group_size"]
    ].drop_duplicates()
    candidates = representative_weights.rename(
        columns={
            "factor_id": "representative_factor_id",
            "weight": "representative_weight",
        }
    ).merge(
        representative_rows,
        on="representative_factor_id",
        how="inner",
        validate="many_to_one",
    )
    if candidates.empty:
        raise ValueError(
            "no representative weights can be mapped to factor groups"
        )

    selected = candidates.sort_values(
        ["date", "representative_weight", "representative_factor_id"],
        ascending=[True, False, True],
        kind="stable",
    ).drop_duplicates(subset=["date"], keep="first")
    return selected[
        [
            "date",
            "cluster",
            "representative_factor_id",
            "representative_weight",
            "group_size",
        ]
    ].sort_values("date").reset_index(drop=True)


def expand_selected_groups_to_equal_weights(
    selected_groups: pd.DataFrame,
    factor_groups: pd.DataFrame,
) -> pd.DataFrame:
    """把 E5 每日选中的完整因子群展开为群内等权。"""
    required_selected = {"date", "cluster", "representative_factor_id"}
    if missing := required_selected - set(selected_groups.columns):
        raise ValueError(
            f"selected_groups missing columns: {sorted(missing)}"
        )
    if factor_groups.empty:
        raise ValueError("factor_groups cannot be empty")

    members = factor_groups[["cluster", "factor_id"]].drop_duplicates()
    expanded = selected_groups.merge(
        members,
        on="cluster",
        how="left",
        validate="many_to_many",
    )
    expanded["group_size"] = expanded.groupby("date")[
        "factor_id"
    ].transform("size")
    expanded["weight"] = 1.0 / expanded["group_size"].astype(float)
    return expanded[
        [
            "date",
            "cluster",
            "representative_factor_id",
            "factor_id",
            "group_size",
            "weight",
        ]
    ].sort_values(["date", "factor_id"]).reset_index(drop=True)


def select_active_inner_weights(
    selected_groups: pd.DataFrame,
    inner_factor_weights: pd.DataFrame,
) -> pd.DataFrame:
    """按 E6 外层选群结果激活对应内层路由权重。"""
    required_groups = {"date", "cluster", "representative_factor_id"}
    required_weights = {"date", "cluster", "factor_id", "weight"}
    if missing := required_groups - set(selected_groups.columns):
        raise ValueError(
            f"selected_groups missing columns: {sorted(missing)}"
        )
    if missing := required_weights - set(inner_factor_weights.columns):
        raise ValueError(
            f"inner_factor_weights missing columns: {sorted(missing)}"
        )

    selected = selected_groups[
        ["date", "cluster", "representative_factor_id"]
    ].drop_duplicates(["date"])
    active = selected.merge(
        inner_factor_weights,
        on=["date", "cluster"],
        how="left",
        validate="one_to_many",
    )
    if active["factor_id"].isna().any():
        missing_dates = active.loc[
            active["factor_id"].isna(),
            "date",
        ].astype(str)
        raise ValueError(
            f"inner router weights are missing for dates: "
            f"{missing_dates.tolist()}"
        )

    totals = active.groupby("date")["weight"].transform("sum")
    if totals.le(0).any() or ~np.isfinite(totals).all():
        raise ValueError(
            "inner router weights must have a positive finite daily sum"
        )
    active["weight"] = active["weight"] / totals
    return active[
        [
            "date",
            "cluster",
            "representative_factor_id",
            "factor_id",
            "weight",
        ]
    ].sort_values(["date", "factor_id"]).reset_index(drop=True)


def _map_weights_to_clusters(
    representative_weights: pd.DataFrame,
    factor_groups: pd.DataFrame,
) -> pd.DataFrame:
    """把外层代表因子权重映射到所属因子群。"""
    required_weights = {"date", "factor_id", "weight"}
    required_groups = {
        "cluster",
        "factor_id",
        "representative_factor_id",
        "is_representative",
        "group_size",
    }
    if missing := required_weights - set(representative_weights.columns):
        raise ValueError(
            f"representative_weights missing columns: {sorted(missing)}"
        )
    if missing := required_groups - set(factor_groups.columns):
        raise ValueError(
            f"factor_groups missing columns: {sorted(missing)}"
        )

    representative_rows = factor_groups[
        factor_groups["is_representative"]
    ][
        ["cluster", "representative_factor_id", "group_size"]
    ].drop_duplicates()
    candidates = representative_weights.rename(
        columns={
            "factor_id": "representative_factor_id",
            "weight": "representative_weight",
        }
    ).merge(
        representative_rows,
        on="representative_factor_id",
        how="inner",
        validate="many_to_one",
    )
    if candidates.empty:
        raise ValueError(
            "no representative weights can be mapped to factor groups"
        )
    candidates["date"] = pd.to_datetime(candidates["date"], errors="coerce")
    candidates["cluster"] = candidates["cluster"].astype(int)
    candidates["representative_weight"] = pd.to_numeric(
        candidates["representative_weight"],
        errors="coerce",
    )
    return candidates


def apply_min_stay_primary_cluster(
    ranked_daily: pd.DataFrame,
    min_stay: int = E6_MIN_STAY_DAYS,
) -> pd.DataFrame:
    """按最短停留约束选择每日主簇；未满停留天数则忽略瞬时最优。"""
    if min_stay <= 0:
        raise ValueError("min_stay must be positive")
    required = {
        "date",
        "cluster",
        "representative_factor_id",
        "representative_weight",
        "group_size",
    }
    if missing := required - set(ranked_daily.columns):
        raise ValueError(
            f"ranked_daily missing columns: {sorted(missing)}"
        )
    if ranked_daily.empty:
        raise ValueError("ranked_daily cannot be empty")

    frame = ranked_daily.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["cluster"] = frame["cluster"].astype(int)
    frame["representative_weight"] = pd.to_numeric(
        frame["representative_weight"],
        errors="coerce",
    )
    rows: list[dict[str, Any]] = []
    held_cluster: int | None = None
    days_held = 0
    for date, day in frame.groupby("date", sort=True):
        day_sorted = day.sort_values(
            ["representative_weight", "representative_factor_id"],
            ascending=[False, True],
            kind="stable",
        )
        instant_best = int(day_sorted.iloc[0]["cluster"])
        if held_cluster is None:
            held_cluster = instant_best
            days_held = 1
        elif days_held < min_stay:
            days_held += 1
        elif instant_best != held_cluster:
            held_cluster = instant_best
            days_held = 1
        else:
            days_held += 1
        held_rows = day_sorted[day_sorted["cluster"].eq(held_cluster)]
        if held_rows.empty:
            raise ValueError(
                f"held cluster {held_cluster} is missing on {date}"
            )
        held = held_rows.iloc[0]
        rows.append(
            {
                "date": date,
                "cluster": int(held["cluster"]),
                "representative_factor_id": str(
                    held["representative_factor_id"]
                ),
                "representative_weight": float(held["representative_weight"]),
                "group_size": int(held["group_size"]),
                "days_held": days_held,
            }
        )
    return pd.DataFrame(rows)


def select_soft_top2_groups(
    representative_weights: pd.DataFrame,
    factor_groups: pd.DataFrame,
    min_stay: int = E6_MIN_STAY_DAYS,
    soft_top_k: int = E6_SOFT_TOP_K,
) -> pd.DataFrame:
    """E6：主簇最短停留后再按 soft_top_k 决定是否混合第二簇。"""
    candidates = _map_weights_to_clusters(
        representative_weights,
        factor_groups,
    )
    primary = apply_min_stay_primary_cluster(candidates, min_stay=min_stay)
    primary_by_date = {
        pd.Timestamp(row["date"]): row
        for row in primary.to_dict(orient="records")
    }

    rows: list[dict[str, Any]] = []
    for date, day in candidates.groupby("date", sort=True):
        primary_row = primary_by_date[pd.Timestamp(date)]
        primary_cluster = int(primary_row["cluster"])
        day_sorted = day.sort_values(
            ["representative_weight", "representative_factor_id"],
            ascending=[False, True],
            kind="stable",
        )
        primary_day = day_sorted[day_sorted["cluster"].eq(primary_cluster)]
        if primary_day.empty:
            raise ValueError(
                f"primary cluster {primary_cluster} is missing on {date}"
            )
        primary_day_row = primary_day.iloc[0]
        others = day_sorted[day_sorted["cluster"].ne(primary_cluster)]
        primary_weight = float(primary_day_row["representative_weight"])
        if others.empty or soft_top_k <= 1:
            rows.append(
                {
                    "date": date,
                    "cluster": primary_cluster,
                    "representative_factor_id": str(
                        primary_day_row["representative_factor_id"]
                    ),
                    "representative_weight": primary_weight,
                    "group_size": int(primary_day_row["group_size"]),
                    "group_weight": 1.0,
                    "is_primary": True,
                }
            )
            continue

        second = others.iloc[0]
        second_weight = float(second["representative_weight"])
        total = primary_weight + second_weight
        if not np.isfinite(total) or total <= 0:
            primary_group_weight = 1.0
            second_group_weight = 0.0
        else:
            primary_group_weight = primary_weight / total
            second_group_weight = second_weight / total
        rows.append(
            {
                "date": date,
                "cluster": primary_cluster,
                "representative_factor_id": str(
                    primary_day_row["representative_factor_id"]
                ),
                "representative_weight": primary_weight,
                "group_size": int(primary_day_row["group_size"]),
                "group_weight": float(primary_group_weight),
                "is_primary": True,
            }
        )
        if second_group_weight > 0:
            rows.append(
                {
                    "date": date,
                    "cluster": int(second["cluster"]),
                    "representative_factor_id": str(
                        second["representative_factor_id"]
                    ),
                    "representative_weight": second_weight,
                    "group_size": int(second["group_size"]),
                    "group_weight": float(second_group_weight),
                    "is_primary": False,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["date", "is_primary", "cluster"],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)


def blend_inner_weights_with_group_weights(
    selected_groups: pd.DataFrame,
    inner_factor_weights: pd.DataFrame,
) -> pd.DataFrame:
    """按 E6 软选群的 group_weight 混合各簇内层权重。"""
    required_groups = {
        "date",
        "cluster",
        "representative_factor_id",
        "group_weight",
    }
    required_weights = {"date", "cluster", "factor_id", "weight"}
    if missing := required_groups - set(selected_groups.columns):
        raise ValueError(
            f"selected_groups missing columns: {sorted(missing)}"
        )
    if missing := required_weights - set(inner_factor_weights.columns):
        raise ValueError(
            f"inner_factor_weights missing columns: {sorted(missing)}"
        )

    selected = selected_groups[
        ["date", "cluster", "representative_factor_id", "group_weight"]
    ].copy()
    selected["date"] = pd.to_datetime(selected["date"], errors="coerce")
    selected["cluster"] = selected["cluster"].astype(int)
    selected["group_weight"] = pd.to_numeric(
        selected["group_weight"],
        errors="coerce",
    )
    inner = inner_factor_weights.copy()
    inner["date"] = pd.to_datetime(inner["date"], errors="coerce")
    inner["cluster"] = inner["cluster"].astype(int)
    inner["weight"] = pd.to_numeric(inner["weight"], errors="coerce")
    active = selected.merge(
        inner,
        on=["date", "cluster"],
        how="left",
        validate="many_to_many",
    )
    if active["factor_id"].isna().any():
        missing_dates = active.loc[
            active["factor_id"].isna(),
            "date",
        ].astype(str)
        raise ValueError(
            f"inner router weights are missing for dates: "
            f"{missing_dates.tolist()}"
        )
    active["weight"] = active["weight"] * active["group_weight"]
    totals = active.groupby("date")["weight"].transform("sum")
    if totals.le(0).any() or ~np.isfinite(totals).all():
        raise ValueError(
            "blended inner weights must have a positive finite daily sum"
        )
    active["weight"] = active["weight"] / totals
    return active[
        [
            "date",
            "cluster",
            "representative_factor_id",
            "factor_id",
            "weight",
        ]
    ].sort_values(["date", "factor_id"]).reset_index(drop=True)


def merge_small_clusters(
    factor_groups: pd.DataFrame,
    centers: np.ndarray,
    min_size: int = E6_MIN_CLUSTER_SIZE,
) -> tuple[pd.DataFrame, dict[int, int]]:
    """把人数过小的簇并入最近的足够大簇，供 E5/E6 运行时使用。"""
    required = {
        "cluster",
        "factor_id",
        "representative_factor_id",
        "is_representative",
        "group_size",
    }
    if missing := required - set(factor_groups.columns):
        raise ValueError(
            f"factor_groups missing columns: {sorted(missing)}"
        )
    if min_size <= 1:
        raise ValueError("min_size must be greater than 1")
    if factor_groups.empty:
        raise ValueError("factor_groups cannot be empty")

    frame = factor_groups.copy()
    frame["cluster"] = frame["cluster"].astype(int)
    started_in = frame["cluster"].copy()
    centers_array = np.asarray(centers, dtype=float)
    if centers_array.ndim != 2:
        raise ValueError("centers must be a 2-D array")
    current_centers: dict[int, np.ndarray] = {}
    for cluster_id in frame["cluster"].unique().tolist():
        cluster_id = int(cluster_id)
        if cluster_id < 0 or cluster_id >= len(centers_array):
            raise ValueError(f"cluster {cluster_id} has no matching center")
        current_centers[cluster_id] = centers_array[cluster_id].copy()

    def cluster_sizes() -> pd.Series:
        return (
            frame.groupby("cluster")["factor_id"]
            .nunique()
            .astype(int)
        )

    def absorb(source: int, target: int) -> None:
        if source == target:
            return
        sizes = cluster_sizes()
        size_source = int(sizes.loc[source])
        size_target = int(sizes.loc[target])
        target_rep = frame.loc[
            frame["cluster"].eq(target) & frame["is_representative"],
            "representative_factor_id",
        ]
        if target_rep.empty:
            raise ValueError(f"target cluster {target} has no representative")
        representative = str(target_rep.iloc[0])
        current_centers[target] = (
            current_centers[target] * size_target
            + current_centers[source] * size_source
        ) / float(size_source + size_target)
        del current_centers[source]
        members = frame["cluster"].eq(source)
        frame.loc[members, "cluster"] = target
        frame.loc[members, "representative_factor_id"] = representative
        frame.loc[members, "is_representative"] = False

    while True:
        sizes = cluster_sizes()
        if len(sizes) <= 1:
            break
        small = sorted(
            int(cluster)
            for cluster, size in sizes.items()
            if int(size) < min_size
        )
        large = sorted(
            int(cluster)
            for cluster, size in sizes.items()
            if int(size) >= min_size
        )
        if not small:
            break
        if large:
            source = min(small, key=lambda cluster: (int(sizes.loc[cluster]), cluster))
            target = min(
                large,
                key=lambda cluster: (
                    float(
                        np.linalg.norm(
                            current_centers[source] - current_centers[cluster]
                        )
                    ),
                    cluster,
                ),
            )
            absorb(source, target)
            continue
        ids = sorted(int(cluster) for cluster in sizes.index)
        best: tuple[float, int, int, int] | None = None
        for index, left in enumerate(ids):
            for right in ids[index + 1 :]:
                distance = float(
                    np.linalg.norm(
                        current_centers[left] - current_centers[right]
                    )
                )
                size_left = int(sizes.loc[left])
                size_right = int(sizes.loc[right])
                if size_right > size_left or (
                    size_right == size_left and right < left
                ):
                    source, target = left, right
                else:
                    source, target = right, left
                candidate = (distance, min(left, right), source, target)
                if best is None or candidate[:2] < best[:2]:
                    best = candidate
        if best is None:
            break
        absorb(best[2], best[3])

    frame["group_size"] = (
        frame.groupby("cluster")["factor_id"].transform("nunique").astype(int)
    )
    living = sorted(int(cluster) for cluster in frame["cluster"].unique())
    compact = {old: index for index, old in enumerate(living)}
    frame["cluster"] = frame["cluster"].map(compact).astype(int)
    merge_map: dict[int, int] = {}
    for original in started_in.unique().tolist():
        destinations = frame.loc[
            started_in.eq(original),
            "cluster",
        ].unique()
        if len(destinations) != 1:
            raise ValueError(
                f"original cluster {int(original)} split during merge"
            )
        merge_map[int(original)] = int(destinations[0])
    return (
        frame.sort_values(
            ["cluster", "is_representative", "factor_id"],
            ascending=[True, False, True],
            kind="stable",
        ).reset_index(drop=True),
        merge_map,
    )


def save_factor_groups(
    run_dir: Path,
    factor_groups: pd.DataFrame,
) -> None:
    """同时保存因子群总表及每个簇的独立成员清单。"""
    factor_groups.to_parquet(
        run_dir / "factor_groups.parquet",
        index=False,
    )
    groups_dir = run_dir / "factor_groups"
    groups_dir.mkdir()
    all_groups: list[dict[str, Any]] = []
    for cluster, group in factor_groups.groupby("cluster", sort=True):
        representative = str(
            group["representative_factor_id"].iloc[0]
        )
        payload = {
            "cluster": int(cluster),
            "representative_factor_id": representative,
            "group_size": int(len(group)),
            "factor_ids": group["factor_id"].astype(str).tolist(),
        }
        write_json(
            groups_dir / f"cluster_{int(cluster):03d}.json",
            payload,
        )
        all_groups.append(payload)
    write_json(run_dir / "factor_groups.json", {"groups": all_groups})
