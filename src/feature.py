import numpy as np
import shap
import torch
from torch import nn


class MaliciousProb(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return torch.softmax(self.model(x), dim=1)[:, 1:]


def shap_values(model, background, x):
    explainer = shap.GradientExplainer(MaliciousProb(model), background)
    return np.asarray(explainer.shap_values(x)).reshape(x.shape)


def evasion_potential(values, y):
    """EP = SHAP value range x positive activation rate on malicious samples."""
    spread = values.max(0) - values.min(0)
    activation = (values[y == 1] > 0).mean(0)
    return spread * activation


def partition_features(ep, rho=0.5):
    """Top-rho features by EP form the MFS, the rest form the NFS."""
    order = np.argsort(-ep, kind="stable")
    k = int(rho * len(ep))
    return np.sort(order[:k]), np.sort(order[k:])
