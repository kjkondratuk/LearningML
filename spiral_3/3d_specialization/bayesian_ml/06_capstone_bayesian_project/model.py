"""Bayesian model implementations."""
import torch.nn as nn

class BayesianCNN(nn.Module):
    def __init__(self, num_classes: int = 7):
        raise NotImplementedError
    def forward(self, x):
        raise NotImplementedError
