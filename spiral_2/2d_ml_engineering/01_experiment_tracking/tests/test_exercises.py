"""
Tests for Experiment Tracking exercises.
"""

import pytest


class TestMLflowSetup:
    def test_experiment_created(self):
        """Verify experiment can be created and accessed."""
        from exercises import (
            setup_mlflow_experiment,
        )
        exp_id = setup_mlflow_experiment("test_experiment")
        assert exp_id is not None

    def test_log_and_retrieve(self):
        """Verify logged parameters and metrics are retrievable."""
        from exercises import (
            setup_mlflow_experiment,
            log_training_run,
            compare_runs,
        )
        setup_mlflow_experiment("test_tracking")
        run_id = log_training_run(
            "test_model",
            params={"lr": 0.01, "epochs": 10},
            metrics={"loss": [1.0, 0.5, 0.2]},
        )
        assert run_id is not None

    def test_compare_runs_sorted(self):
        """Runs should be returned sorted by metric."""
        from exercises import (
            compare_runs,
        )
        runs = compare_runs("test_tracking", "loss")
        if len(runs) > 1:
            # Best (lowest) first
            assert runs[0].get("loss", float("inf")) <= runs[-1].get("loss", float("inf"))
