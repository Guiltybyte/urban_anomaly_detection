"""
Graph Feature Autoencoder for Prediction implementations
"""
from torch_geometric.nn import SAGEConv, GraphConv, GCNConv
import torch.nn as nn
import torch
from torch.nn import Linear as Lin

from models.End_to_End.layers import FeatGraphConv


class FAE_FeatGraphConv(nn.Module):
    """
    Graph Feature Autoencoder Network, using the custom
    FeatGraphConv layer as introduced in the same paper.
    """
    def __init__(self, in_channels):
        super(FAE_FeatGraphConv, self).__init__()
        self.conv1 = FeatGraphConv(in_channels, 64, 64, aggr='mean')
        self.conv2 = FeatGraphConv(64, 32, 32, aggr='mean')
        # Outputs single dimension because a seperate model is being trained for each feature (column)
        self.lin = Lin(32, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        return self.lin(x)
