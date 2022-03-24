## Utility
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

## ML
import torch 
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric_temporal.nn.recurrent import A3TGCN2
from typing import Final

## Dataset
from data_loader.sumo_incident_dataset import SumoIncidentDataset
from torch_geometric.loader import DataLoader
from torch_geometric_temporal.signal import temporal_signal_split

# Metrics
from torch_geometric.utils.metric import accuracy 

DEVICE: Final = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE: Final[int] = 4
NUM_TIMESTEPS: Final[int] = 12
SHUFFLE: Final[bool] = True

## Dataset setup
loader = SumoIncidentDataset('data', 'SUMO_incident', NUM_TIMESTEPS)
data_set = loader.get_data()
train_dataset, test_dataset = temporal_signal_split(data_set, train_ratio=0.9)

train_input = np.array(train_dataset.features) # (27399, 207, 2, 12)
train_target = np.array(train_dataset.targets) # (27399, 207, 12)
train_x_tensor = torch.from_numpy(train_input).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
train_dataset_new = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset_new, batch_size=BATCH_SIZE, shuffle=SHUFFLE,drop_last=True)

test_input = np.array(test_dataset.features) # (, 207, 2, 12)
test_target = np.array(test_dataset.targets) # (, 207, 12)
test_x_tensor = torch.from_numpy(test_input).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
test_dataset_new = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)
test_loader = torch.utils.data.DataLoader(test_dataset_new, batch_size=BATCH_SIZE, shuffle=SHUFFLE, drop_last=True)


class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, periods, batch_size):
        super(TemporalGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN2(in_channels=node_features,  out_channels=32, periods=periods,batch_size=batch_size) 
        self.tgnn = A3TGCN2(in_channels=node_features,  out_channels=1, periods=periods,batch_size=batch_size) 
        # TODO change it to predicting the last rather than the first
        # Only predict labels for current time (hence the single output channel)
        # self.linear = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, edge_index) # x [b, 207, 5, 12]  returns h [b, 207, 12]
        h = F.relu(h) 
        # h = self.linear(h)
        h = torch.reshape(h, (h.size(0), h.size(1)))
        print(h)
        return torch.sigmoid(h)


# Create model and optimizers
model = TemporalGNN(node_features=loader.num_node_features, periods=NUM_TIMESTEPS, batch_size=BATCH_SIZE).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.BCELoss()


print('Net\'s state_dict:')
total_param = 0
for param_tensor in model.state_dict():
    print(param_tensor, '\t', model.state_dict()[param_tensor].size())
    total_param += np.prod(model.state_dict()[param_tensor].size())
print('Net\'s total params:', total_param)
#--------------------------------------------------
print('Optimizer\'s state_dict:')  # If you notice here the Attention is a trainable parameter
for var_name in optimizer.state_dict():
    print(var_name, '\t', optimizer.state_dict()[var_name])


# Loading the graph once because it's a static graph

for snapshot in train_dataset:
    static_edge_index = snapshot.edge_index.to(DEVICE)
    break;

# Training the model 
model.train()
for epoch in tqdm(range(30)):
    step = 0
    loss_list = []
    for encoder_inputs, labels in train_loader:
        # Y_hat is a prediction of the next state perhaps I can treat y like a feature
        y_hat = model(encoder_inputs, static_edge_index)         # Get model predictions
        loss = loss_fn(y_hat, labels)

        loss.backward()
        optimizer.step()
        # optimizer.zero_grad()
        step += 1
        loss_list.append(loss.item())
        if step % 100 == 0 :
            print(sum(loss_list)/len(loss_list))
    print("Epoch {} train RMSE: {:.4f}".format(epoch, sum(loss_list)/len(loss_list)))
