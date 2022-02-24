import torch
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from models.mp_gcn_conv import GCNConv
from typing import Final


class GCN(torch.nn.Module):
    """
    Two layer, Graph convolutional Neural Network
    which returns a softmax distribution 
    describing the probability of each node having each
    class.
    """
    # for project: in_channels should be num_node_features,
    #              out_channels should be num_classes
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, out_channels)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index) # what the fuck is the issue here
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        # x = global_add_pool(x, batch)

        return F.log_softmax(x, dim=1)
