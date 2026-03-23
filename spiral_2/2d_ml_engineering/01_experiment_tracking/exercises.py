"""
Experiment Tracking Exercises -- Spiral 2, Phase 2D, Module 01
"""


def setup_mlflow_experiment(experiment_name: str):
    """Create an MLflow experiment. Return experiment ID.

    Args:
        experiment_name: name for the experiment

    Returns:
        experiment_id: string ID
    """
    raise NotImplementedError


def log_training_run(
    model_name: str,
    params: dict,
    metrics: dict,
    artifacts_dir: str = None,
) -> str:
    """Log a complete training run to MLflow.

    Log: hyperparameters, metrics at each epoch, model artifacts.

    Args:
        model_name: name of the model
        params: hyperparameter dict
        metrics: dict of metric_name -> list of values per epoch
        artifacts_dir: optional directory of artifacts to log

    Returns:
        run_id: MLflow run ID
    """
    raise NotImplementedError


def compare_runs(experiment_name: str, metric_name: str) -> list[dict]:
    """Query MLflow for all runs, return sorted by metric.

    Args:
        experiment_name: which experiment
        metric_name: which metric to sort by

    Returns:
        list of run dicts sorted by metric (best first)
    """
    raise NotImplementedError


def log_model_with_signature(model, X_sample, y_sample, model_name: str) -> str:
    """Log a model with its input/output signature.

    Args:
        model: trained model object
        X_sample: example input
        y_sample: example output
        model_name: registered model name

    Returns:
        model_uri: URI of the logged model
    """
    raise NotImplementedError


def create_experiment_report(experiment_name: str) -> dict:
    """Generate a summary of an experiment.

    Returns:
        dict with 'best_run', 'param_distributions', 'metric_summary'
    """
    raise NotImplementedError
