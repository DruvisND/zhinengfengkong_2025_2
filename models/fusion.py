"""Gated Fusion Layer — 三路模型输出动态加权融合"""

import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    """
    注意力门控融合层。

    将 LightGBM 概率、GraphSAGE Embedding、Transformer CLS Embedding
    动态加权融合后接 MLP 输出最终欺诈概率。
    """

    def __init__(
        self,
        lgb_dim: int = 1,
        gnn_dim: int = 64,
        seq_dim: int = 32,
        hidden_dim: int = 64,
    ):
        super().__init__()
        total_dim = lgb_dim + gnn_dim + seq_dim

        self.gate = nn.Sequential(
            nn.Linear(total_dim, total_dim),
            nn.Sigmoid(),
        )

        self.mlp = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, lgb_prob, gnn_emb, seq_emb):
        """
        Parameters
        ----------
        lgb_prob : (batch, 1) LightGBM 预测概率
        gnn_emb  : (batch, gnn_dim) GraphSAGE 节点 Embedding
        seq_emb  : (batch, seq_dim) Transformer CLS Embedding
        """
        combined = torch.cat([lgb_prob, gnn_emb, seq_emb], dim=1)
        gate_weights = self.gate(combined)
        gated = combined * gate_weights
        return self.mlp(gated).squeeze(1)
