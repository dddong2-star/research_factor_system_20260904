from pathlib import Path

from research_pipeline.step3_clustering.cluster import run_cluster
from research_pipeline.step4_experiments.run import run_experiments
from research_pipeline.step4_experiments.specs import build_experiment_specs
from research_pipeline.tests.helpers import write_experiment_inputs


def test_build_experiment_specs_has_incremental_four_level_design():
    specs = build_experiment_specs(["f1", "f2"], test_start="2024-01-03")

    assert [item["experiment_id"] for item in specs] == ["E1", "E2", "E3", "E4"]
    assert specs[0]["components"] == ["kmeans", "equal_weight", "cost_backtest"]
    assert "macoe" in specs[1]["components"]
    assert "pacoe" in specs[2]["components"]
    assert {"macoe", "pacoe", "gumbel_topk", "turnover_penalty"}.issubset(
        specs[3]["components"]
    )
    assert all(item["factor_ids"] == ["f1", "f2"] for item in specs)


def test_run_experiments_writes_four_level_outputs(tmp_path):
    inputs = write_experiment_inputs(tmp_path)
    cluster = run_cluster(
        data_root=inputs["data_root"],
        values_run=inputs["values_run"],
        state_run=inputs["state_run"],
        output_root=tmp_path / "clusters",
        snapshot_id="snap",
        test_start="2024-01-05",
        horizon=1,
        n_clusters=1,
    )

    result = run_experiments(
        cluster_run=cluster["run_dir"],
        output_root=tmp_path / "runs",
        experiments="E1,E2,E3,E4",
        top_n=1,
        epochs=1,
        factor_top_k=1,
    )

    suite = result["e1_e4"]
    assert [item["experiment_id"] for item in suite["experiments"]] == [
        "E1",
        "E2",
        "E3",
        "E4",
    ]
    run_dir = Path(suite["run_dir"])
    assert all(
        (run_dir / item["experiment_id"] / "metrics.json").exists()
        for item in suite["experiments"]
    )
    assert (run_dir / "daily_factor_performance.parquet").exists()
    assert (run_dir / "factor_clusters.parquet").exists()
    assert (run_dir / "experiment_suite.json").exists()
    assert (run_dir / "E2" / "model.pt").exists()
    assert not (run_dir / "E1" / "model.pt").exists()
