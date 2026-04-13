"""Transformer 路径 — 用户交易行为序列编码器"""

import torch
import torch.nn as nn


class BehaviorTransformer(nn.Module):
    """
    用户交易行为序列编码器。

    输入: (batch, seq_len, feat_dim) 的交易序列
    输出: (batch, feat_dim) 的 CLS token Embedding
    """

    def __init__(
        self,
        feat_dim: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        max_len: int = 50,
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=n_heads,
            dim_feedforward=feat_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pos_emb = nn.Embedding(max_len + 1, feat_dim)  # +1 for CLS
        self.cls_token = nn.Parameter(torch.randn(1, 1, feat_dim))
        self.input_proj = nn.Linear(feat_dim, feat_dim)

    def forward(self, x, mask=None):
        """
        Parameters
        ----------
        x : Tensor of shape (batch, seq_len, feat_dim)
        mask : optional padding mask
        """
        B, S, _ = x.shape
        x = self.input_proj(x)

        pos = torch.arange(S + 1, device=x.device).unsqueeze(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_emb(pos)

        out = self.encoder(x, src_key_padding_mask=mask)
        return out[:, 0, :]  # CLS token


def build_sequences(df, group_col="card1", seq_len=50, feat_cols=None):
    """
    按 group_col 分组构建交易序列。

    Returns: dict mapping group_id -> (seq_features, label)
    """
    if feat_cols is None:
        feat_cols = ["amt_log", "hour", "weekday"]

    sequences = {}
    for gid, group in df.groupby(group_col):
        group = group.sort_values("TransactionDT")
        feats = group[feat_cols].values[-seq_len:]
        label = group["isFraud"].values[-1]

        if len(feats) < seq_len:
            pad = torch.zeros(seq_len - len(feats), len(feat_cols))
            feats = torch.cat([pad, torch.tensor(feats, dtype=torch.float32)])
        else:
            feats = torch.tensor(feats, dtype=torch.float32)

        sequences[gid] = (feats, label)

    return sequences
