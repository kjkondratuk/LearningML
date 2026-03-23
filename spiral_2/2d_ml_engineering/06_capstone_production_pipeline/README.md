# Module 06: Capstone -- Production Pipeline

> Take the best model from Phase 2B or 2C and build a production-quality training
> pipeline with all the engineering practices from this phase.

## Requirements

1. **`pipeline.py`**: Main training pipeline with:
   - Experiment tracking (MLflow)
   - Hyperparameter tuning (Optuna)
   - Proper data pipeline (DataLoader with augmentation)
   - Model checkpointing and early stopping
   - Learning rate scheduling and gradient clipping
   - Mixed precision training

2. **`config.yaml`**: All hyperparameters in YAML (no hardcoded values).

3. **Reproducibility**: Seed everything, log all configs, save commit hash.

4. **Comprehensive logging**: Loss curves, gradient norms, LR schedule, time per epoch.

5. **"Run training" script**: Takes a config file and produces a fully tracked experiment.

6. **Final notebook**: Load best model from MLflow, evaluate, produce a model card.

## Acceptance Criteria

- `python pipeline.py --config config.yaml` runs end-to-end
- Experiment appears in MLflow with all metrics and artifacts
- Model achieves reasonable accuracy on the chosen dataset
- All tests pass
