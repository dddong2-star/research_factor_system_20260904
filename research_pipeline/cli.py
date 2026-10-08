"""唯一命令入口：按 step1–step4 分发研究流水线。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .step1_snapshot.export import ExportSources, export_snapshot
from .step1_snapshot.validate import validate_snapshot
from .step2_features.conditional import run_conditional_evaluation
from .step2_features.select import run_selection
from .step2_features.states import run_state_analysis
from .step2_features.values import materialize_factor_values
from .step3_clustering.cluster import run_cluster
from .step4_experiments.run import run_experiments


def configure_utf8_output() -> None:
    """统一 Windows 终端输出编码，避免中文乱码。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def _load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        value = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "YAML config requires PyYAML; use a JSON config instead"
            ) from exc
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError("config must contain a mapping")
    return value


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="独立因子研究流水线")
    steps = parser.add_subparsers(dest="step", required=True)

    step1 = steps.add_parser("step1", help="导出或校验不可变研究快照")
    step1_actions = step1.add_subparsers(dest="action", required=True)
    export_parser = step1_actions.add_parser("export", help="导出一份不可变研究快照")
    export_parser.add_argument("--config", required=True, help="YAML 或 JSON 导出配置")
    export_parser.add_argument("--snapshot-id")
    export_parser.add_argument("--overwrite", action="store_true")
    validate_parser = step1_actions.add_parser("validate", help="校验研究快照")
    validate_parser.add_argument("--data-root", default="research_data")
    validate_parser.add_argument("--snapshot-id")

    step2 = steps.add_parser("step2", help="生成市场状态、因子值及可选条件评价")
    step2_actions = step2.add_subparsers(dest="action", required=True)
    state_parser = step2_actions.add_parser("states", help="由快照生成规则市场状态")
    state_parser.add_argument("--data-root", default="research_data")
    state_parser.add_argument("--runs-root", default="research_runs")
    state_parser.add_argument("--snapshot-id")
    state_parser.add_argument("--volatility-window", type=int, default=20)
    state_parser.add_argument("--trend-threshold", type=float, default=0.002)
    state_parser.add_argument("--volatility-threshold", type=float, default=0.03)
    values_parser = step2_actions.add_parser("values", help="按表达式生成分区因子值")
    values_parser.add_argument("--data-root", default="research_data")
    values_parser.add_argument("--runs-root", default="research_runs")
    values_parser.add_argument("--snapshot-id")
    values_parser.add_argument("--factor-id", action="append", dest="factor_ids")
    values_parser.add_argument("--max-factors", type=int)
    conditional_parser = step2_actions.add_parser(
        "conditional",
        help="按市场状态和时间切分评价因子值",
    )
    conditional_parser.add_argument("--data-root", default="research_data")
    conditional_parser.add_argument("--values-run", required=True)
    conditional_parser.add_argument("--state-run", required=True)
    conditional_parser.add_argument("--runs-root", default="research_runs")
    conditional_parser.add_argument("--snapshot-id")
    conditional_parser.add_argument("--horizon", type=int, default=20)
    conditional_parser.add_argument("--max-factors", type=int)
    conditional_parser.add_argument("--train-ratio", type=float, default=0.6)
    conditional_parser.add_argument("--valid-ratio", type=float, default=0.2)
    select_parser = step2_actions.add_parser(
        "select",
        help="为一个市场状态和一个时间切分选择因子",
    )
    select_parser.add_argument("--data-root", default="research_data")
    select_parser.add_argument("--conditional-run", required=True)
    select_parser.add_argument("--runs-root", default="research_runs")
    select_parser.add_argument("--snapshot-id")
    select_parser.add_argument("--state-id", required=True)
    select_parser.add_argument("--category")
    select_parser.add_argument(
        "--split",
        choices=["train", "valid", "test"],
        default="train",
    )
    select_parser.add_argument("--top-n", type=int, default=10)
    select_parser.add_argument("--min-sample-count", type=int, default=10)
    select_parser.add_argument("--correlation-threshold", type=float, default=0.8)

    step3 = steps.add_parser("step3", help="对训练期行为画像聚类并保存因子群")
    step3_actions = step3.add_subparsers(dest="action", required=True)
    cluster_parser = step3_actions.add_parser(
        "cluster",
        help="生成供 E1–E6 共用的聚类产物",
    )
    cluster_parser.add_argument("--data-root", default="research_data")
    cluster_parser.add_argument("--values-run", required=True)
    cluster_parser.add_argument("--state-run", required=True)
    cluster_parser.add_argument("--output-root", default="research_runs")
    cluster_parser.add_argument("--snapshot-id")
    cluster_parser.add_argument("--factor-id", action="append", dest="factor_ids")
    cluster_parser.add_argument("--max-factors", type=int)
    cluster_parser.add_argument("--test-start")
    cluster_parser.add_argument("--horizon", type=int, default=20)
    cluster_parser.add_argument("--n-clusters", type=int, default=20)
    cluster_parser.add_argument("--random-state", type=int, default=42)

    step4 = steps.add_parser("step4", help="读取聚类产物并运行 E1–E6")
    step4_actions = step4.add_subparsers(dest="action", required=True)
    run_parser = step4_actions.add_parser("run", help="运行指定的 E1–E6 实验")
    run_parser.add_argument("--cluster-run", required=True)
    run_parser.add_argument("--output-root", default="research_runs")
    run_parser.add_argument(
        "--experiments",
        default="E1,E2,E3,E4,E5,E6",
        help="逗号分隔的实验编号，默认全部六级",
    )
    run_parser.add_argument("--top-n", type=int, default=20)
    run_parser.add_argument("--factor-top-k", type=int)
    run_parser.add_argument("--fee-rate", type=float, default=0.001)
    run_parser.add_argument("--slippage-rate", type=float, default=0.001)
    run_parser.add_argument("--rolling-window", type=int, default=40)
    run_parser.add_argument("--epochs", type=int, default=10)
    run_parser.add_argument("--random-state", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_utf8_output()
    args = build_parser().parse_args(argv)
    if args.step == "step1" and args.action == "export":
        config = _load_config(args.config)
        sources = ExportSources(
            market_panel=config["market_panel"],
            pool_dir=config["pool_dir"],
            factor_runs_root=config.get("factor_runs_root"),
            backtests_root=config.get("backtests_root"),
            pool_id=config.get("pool_id"),
        )
        result = export_snapshot(
            sources,
            config.get("output_root", "research_data"),
            args.snapshot_id,
            args.overwrite,
        )
        _print_json(
            {
                "valid": True,
                "snapshot_id": result["snapshot_id"],
                "factor_count": result["factor_count"],
            }
        )
        return 0
    if args.step == "step1" and args.action == "validate":
        result = validate_snapshot(args.data_root, args.snapshot_id)
        _print_json(result)
        return 0 if result["valid"] else 1
    if args.step == "step2" and args.action == "states":
        result = run_state_analysis(
            args.data_root,
            args.runs_root,
            args.snapshot_id,
            args.volatility_window,
            args.trend_threshold,
            args.volatility_threshold,
        )
        _print_json(result)
        return 0
    if args.step == "step2" and args.action == "values":
        result = materialize_factor_values(
            args.data_root,
            args.runs_root,
            args.snapshot_id,
            factor_ids=args.factor_ids,
            max_factors=args.max_factors,
        )
        _print_json(result)
        return 0 if result["failed"] == 0 else 2
    if args.step == "step2" and args.action == "conditional":
        result = run_conditional_evaluation(
            args.data_root,
            args.values_run,
            args.state_run,
            args.runs_root,
            args.snapshot_id,
            args.horizon,
            args.max_factors,
            args.train_ratio,
            args.valid_ratio,
        )
        _print_json(result)
        return 0 if result["failed"] == 0 else 2
    if args.step == "step2" and args.action == "select":
        result = run_selection(
            args.data_root,
            args.conditional_run,
            args.runs_root,
            args.snapshot_id,
            args.state_id,
            args.category,
            args.split,
            args.top_n,
            args.min_sample_count,
            args.correlation_threshold,
        )
        _print_json(result)
        return 0
    if args.step == "step3" and args.action == "cluster":
        result = run_cluster(
            data_root=args.data_root,
            values_run=args.values_run,
            state_run=args.state_run,
            output_root=args.output_root,
            snapshot_id=args.snapshot_id,
            factor_ids=args.factor_ids,
            max_factors=args.max_factors,
            test_start=args.test_start,
            horizon=args.horizon,
            n_clusters=args.n_clusters,
            random_state=args.random_state,
        )
        _print_json(result)
        return 0
    result = run_experiments(
        cluster_run=args.cluster_run,
        output_root=args.output_root,
        experiments=args.experiments,
        top_n=args.top_n,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        rolling_window=args.rolling_window,
        epochs=args.epochs,
        random_state=args.random_state,
        factor_top_k=args.factor_top_k,
    )
    _print_json(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
