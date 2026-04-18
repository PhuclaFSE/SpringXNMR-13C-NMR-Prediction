import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d, LeakyReLU
from torch_geometric.nn import GATv2Conv

class NMRModel(nn.Module):
    def __init__(self, node_in_dim=13, edge_in_dim=10, hidden_dim=256, num_layers=6, heads=8, dropout=0.1):
        super().__init__()
        self.node_embed = Linear(node_in_dim, hidden_dim)
        self.edge_embed = Linear(edge_in_dim, hidden_dim)
        self.edge_updates = nn.ModuleList([Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.conv = nn.ModuleList([
            GATv2Conv(in_channels=hidden_dim, out_channels=hidden_dim//heads,
                      heads=heads, edge_dim=hidden_dim, concat=True, dropout=dropout, add_self_loops=False)
            for _ in range(num_layers)
        ])
        self.bn = nn.ModuleList([BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.out_layer = Sequential(
            Linear(hidden_dim, hidden_dim//2),
            LeakyReLU(),
            Linear(hidden_dim//2, 1)
        )
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.node_embed(x)
        edge_attr = self.edge_embed(edge_attr)
        for i, (conv, bn, edge_upd) in enumerate(zip(self.conv, self.bn, self.edge_updates)):
            h = x
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.leaky_relu(x, 0.01)
            edge_attr = edge_upd(edge_attr)
            if i >= 1:
                x = x + h
            if i % 2 == 1:
                x = F.dropout(x, p=0.1, training=self.training)
        return self.out_layer(x)