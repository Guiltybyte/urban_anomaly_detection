import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Final


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 =
            GCNConv()
