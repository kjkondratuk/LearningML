"""Win-rate evaluation for DPO."""

def compute_win_rate(model, ref_model, test_prompts, evaluator):
    raise NotImplementedError

def compute_diversity(generations: list[str]):
    raise NotImplementedError
