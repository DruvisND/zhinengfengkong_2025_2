"""Focal Loss — 专为极度不均衡欺诈检测设计"""

import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.

    当 alpha=0.25, gamma=2.0 时，对易分类样本（正常交易）降权，
    聚焦难分类的欺诈样本。
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce = nn.functional.binary_cross_entropy_with_logits(
            pred, target, reduction="none"
        )
        pt = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()
