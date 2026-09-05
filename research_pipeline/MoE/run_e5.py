"""E5 命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from .e5_experiment import run_e5_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="运行 E5：E2 选代表因子群，群内全部因子等权"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--values-run", required=True)
    parser.add_argument("--state-run", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--snapshot-id")
    parser.add_argument(
        "--factor-id",
        dest="factor_ids",
        action="append",
        help="限制使用的因子，可重复传入；默认使用全部可用因子",
    )
    parser.add_argument("--max-factors", type=int)
    parser.add_argument("--test-start")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--n-clusters", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--slippage-rate", type=float, default=0.001)
    parser.add_argument("--rolling-window", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # Windows 终端可能沿用非 UTF-8 代码页，显式统一输出编码，避免中文乱码。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    result = run_e5_experiment(
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
        top_n=args.top_n,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        rolling_window=args.rolling_window,
        epochs=args.epochs,
        random_state=args.random_state,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
