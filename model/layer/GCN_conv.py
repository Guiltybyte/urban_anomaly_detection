"""
GCN Convolutional Layer Implementation
"""
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree 

class GCNConv(MessagePassing):
    """
    To intialize and then call this layer:

        conv = GCNConv(16, 32)
        x = conv(x, edge_index)

    where x is the Node feature matrix
    """
    def __init__(self, in_channels, out_channels)
        # add aggregation strategy
        # aggregation is last step of propagation   
        super().__init__(aggr='add') # Step(5)
        self.linearize =
            torch.nn.Linear(in_channels, out_channels)
    
    def forward(self, x, edge_index):
        """
        Forward Pass Operation
        Parameters
        ----------
        x: Node feature Matrix shape=[N, in_channels] 
            (each node will have in_channels features seems fair)
        edge_index: Representation of graph connectivity in COO 
            format shape=[2, E]
        """
        # Step 1: Add self-loops to the adj matrix
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: linearly transform node feature matrix
        x = self.linearize(x)

        # Step 3: Compute normalization
        row, col = edge_index
        # Compute degree of x
        deg = degree(col, x.size(0), dtype=x.dtype) 

        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages
        # NOTE: I think by default propagate only requires the edge_index,
        # the need to pass things like x and norm are defined by whatever
        # parameters we decide to add to message(), update()
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        """
        x_j has shape [E, out_channels]
        Here, x_j denotes a lifted tensor, which contains 
        the source node features of each edge, i.e., the 
        neighbors of each node. Node features can be 
        automatically lifted by appending _i or _j to the variable name
        """
        # Step 4: Normalize node features
        return norm.view(-1 1) * x_j
