import json
import subprocess
import sys

from research_pipeline.cli import build_parser
from research_pipeline.step1_snapshot.validate import validate_snapshot


def test_validate_snapshot_reports_invalid_manifest(tmp_path):
    data_root = tmp_path / "research_data"
    snapshot = data_root / "snapshots" / "bad"
    snapshot.mkdir(parents=True)
    (data_root / "current.txt").write_text("bad\n", encoding="utf-8")
    (snapshot / "manifest.json").write_text(
        json.dumps({"snapshot_id": "bad", "valid": True, "files": []}),
        encoding="utf-8",
    )

    result = validate_snapshot(data_root)

    assert result["valid"] is False
    assert result["errors"]


def test_module_help_exposes_step_commands():
    completed = subprocess.run(
        [sys.executable, "-m", "research_pipeline", "--help"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 0
    assert "step1" in completed.stdout
    assert "step2" in completed.stdout
    assert "step3" in completed.stdout
    assert "step4" in completed.stdout


def test_step2_select_cli_parses_training_selection_options():
    args = build_parser().parse_args(
        [
            "step2",
            "select",
            "--conditional-run",
            "conditional",
            "--state-id",
            "bull",
            "--split",
            "valid",
            "--top-n",
            "5",
        ]
    )

    assert args.step == "step2"
    assert args.action == "select"
    assert args.state_id == "bull"
    assert args.split == "valid"
    assert args.top_n == 5


def test_step3_cluster_cli_parses_research_options():
    args = build_parser().parse_args(
        [
            "step3",
            "cluster",
            "--values-run",
            "values",
            "--state-run",
            "states",
            "--test-start",
            "2024-01-01",
            "--n-clusters",
            "3",
        ]
    )

    assert args.step == "step3"
    assert args.action == "cluster"
    assert args.n_clusters == 3
    assert args.test_start == "2024-01-01"


def test_step4_run_cli_parses_experiment_list():
    args = build_parser().parse_args(
        [
            "step4",
            "run",
            "--cluster-run",
            "cluster",
            "--experiments",
            "E1,E5",
            "--top-n",
            "5",
            "--factor-top-k",
            "2",
        ]
    )

    assert args.step == "step4"
    assert args.action == "run"
    assert args.experiments == "E1,E5"
    assert args.top_n == 5
    assert args.factor_top_k == 2
