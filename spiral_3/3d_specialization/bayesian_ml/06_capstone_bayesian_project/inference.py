"""Bayesian inference methods."""

def predict_with_uncertainty(model, x, method: str = "mc_dropout", n_samples: int = 50):
    raise NotImplementedError

def evaluate_ood_detection(model, in_dist_loader, ood_loader):
    raise NotImplementedError
