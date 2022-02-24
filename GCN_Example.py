# 1. load Dataset
# for this example, the dataset requires no
# transforms or dataloader to be defined
from torch_geometric.datasets import Planetoid
import data_loader.load_csv
import models.gcn
from torch_geometric.loader import DataLoader
# 2A. other imports
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# dataset = Planetoid(root='/tmp/Cora', name='Cora')
data_set = data_loader.load_csv.get_data_list()
train_dataset = data_set[len(data_set) // 10:]
test_dataset = data_set[:len(data_set) // 10]
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=128, shuffle=True)


# 2B. Define a Class representing a GCN including it's forward pass mechanism
# Note: all neural network classes should
# inherit from torch.nn.Module
class GCN(torch.nn.Module):
    """
    Graph Convolutional Neural network with 2 convolutional layers
    """
    def __init__(self):
        super().__init__() # again, required of all NN.
        # first conv layer, the two mandatory args are num input channels and num output
        self.conv1 = GCNConv(3, 16) 
        # 2nd conv layer, note that it has the same amount of input channels as the output channels of conv1
        self.conv2 = GCNConv(self.conv1.out_channels, 2) 


    def forward(self, data):
        """
        Defines the `Forward Pass` through the network,
        i.e. how the network makes predictions
        Note that the non-linearity is not integrated in
        the conv calls and hence needs to be applied afterwards 
        (something which is consistent accross all operators in PyG)
        """
        # ok so it defo has access to the edge_index
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Training 
# Train the model on the training nodes for 200 "epochs"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = train_loader[0].to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=0.01,weight_decay=5e-4)
print("Data: ", data)
# Puts the model in "training mode"
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    print("out shape: ",out.shape)
    print("out : ",out)
    input()

    # The loss shit
    loss = F.nll_loss(out[data.train_mask],
                      data.y[data.train_mask])
    loss.backward()
    optimizer.step()

#4 evaluate
model.eval()
# generate predictions on data using argmax
pred = model(data).argmax(dim=1)
# The Number of correctly predicted labels is simply the amout of predictions which match corresponding actual data lables
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
# Accuracy calculation
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')
