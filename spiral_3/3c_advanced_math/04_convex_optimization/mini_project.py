"""Mini-Project: Proximal Gradient Descent for LASSO

Implement ISTA and FISTA for the LASSO problem and compare convergence.
"""

import torch


def lasso_ista(X: torch.Tensor, y: torch.Tensor, lambda_reg: float, iterations: int = 1000):
    """Solve LASSO with ISTA."""
    raise NotImplementedError


def lasso_fista(X: torch.Tensor, y: torch.Tensor, lambda_reg: float, iterations: int = 1000):
    """Solve LASSO with FISTA (accelerated proximal gradient)."""
    raise NotImplementedError
