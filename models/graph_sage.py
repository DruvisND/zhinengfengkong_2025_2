"""GraphSAGE 路径 — 图神经网络节点 Embedding"""

import torch
import torch.nn as nn


class FraudGraphSAGE(nn.Module):
    """
    2层 GraphSAGE (Mean Aggregator) 用于欺诈关系图的节点表征学习。

    输入节点特征 → 128维隐层 → 64维输出 Embedding。
    """

    def __init__(self, in_feats: int, hidden: int = 128, out_feats: int = 64):
        super().__init__()
        from dgl.nn import SAGEConv

        self.conv1 = SAGEConv(in_feats, hidden, "mean")
        self.conv2 = SAGEConv(hidden, out_feats, "mean")
        self.bn1 = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(0.3)

    def forward(self, g, x):
        h = self.drop(torch.relu(self.bn1(self.conv1(g, x))))
        return self.conv2(g, h)


class GraphSAGETrainer:
    """GraphSAGE 训练器，支持负采样"""

    def __init__(self, model, lr: float = 1e-3, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train_epoch(self, g, features, labels):
        self.model.train()
        g = g.to(self.device)
        features = features.to(self.device)
        labels = labels.to(self.device)

        logits = self.model(g, features)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits.squeeze(), labels.float()
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def get_embeddings(self, g, features):
        self.model.eval()
        g = g.to(self.device)
        features = features.to(self.device)
        return self.model(g, features).cpu().numpy()
