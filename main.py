import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from typing import Final
import data_loader.load_csv
import models.gcn
from data_loader.sumo_incident_dataset import SumoIncidentDataset

data_set = SumoIncidentDataset('data', 'SUMO_incident')
print(f"Num classes: {data_set.num_classes}")
print(f"Num features: {data_set.num_node_features}")

# data_set = data_loader.load_csv.get_data_list()
# train_dataset = data_set[len(data_set) // 10:]
# test_dataset = data_set[:len(data_set) // 10]
# train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
# test_loader  = DataLoader(test_dataset, batch_size=128, shuffle=True)

DEVICE: Final = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
in_channels  = 3
out_channels = 2
model = models.gcn.GCN(in_channels, out_channels).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    model.train() # puts model in training mode
    for data in train_dataset:
        data = data.to(DEVICE)
        optimizer.zero_grad()
        print("data.x", data.x.shape)
        print("data.edge_index", data.edge_index.shape)
        out = model(data.x, data.edge_index)
        print("out: ", out)
        print("out.shape: ", out.shape)
        input()
        # Need to figure out the appropriate loss
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(train_dataset)

train()
