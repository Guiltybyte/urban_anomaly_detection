import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from typing import Final
import data_loader.load_csv
import models.gcn
from data_loader.sumo_incident_dataset import SumoIncidentDataset


# Hyper-Parameters
EPOCHS: Final[int] = 100
BATCH_SIZE: Final[int] = 32 
LEARN_RATE: Final[float] = 0.01

# Setup and test & train dataset
data_set = SumoIncidentDataset('data', 'SUMO_incident')
data_set = data_set.shuffle()
train_dataset = data_set[len(data_set) // 10:]
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset  = data_set[:len(data_set) // 10]
test_loader   = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


# Setup Model
DEVICE: Final = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.gcn.GCN(
            in_channels = train_dataset.num_node_features,
            out_channels = 1
        ).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE, weight_decay=5e-4)

def binary_acc(y_pred, y_actual) -> float:
    y_pred_labels = torch.round(y_pred)
    sum_correct = (y_pred_labels == y_actual).sum().float()
    acc = sum_correct / y_actual.shape[0] 
    return torch.round(acc * 100)


# Training
for e in range(EPOCHS-1):
    epoch_loss = 0 
    epoch_train_acc = 0
    epoch_test_acc = 0

    model.train()
    for data in train_loader:
        data = data.to(DEVICE)
        optimizer.zero_grad()
        pred_out = model(data.x, data.edge_index)
        # Higher weight means more recall, less precision
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(6))
        loss = loss_fn(
                pred_out, 
                data.y.type(torch.FloatTensor)
                )

        # loss = F.binary_cross_entropy(
        #         pred_out, 
        #         data.y.type(torch.FloatTensor),
        #         )
        acc = binary_acc(
                torch.sigmoid(pred_out), data.y.type(torch.FloatTensor)
                ) 
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_train_acc += acc.item()

    # Evaluation - testing
    model.eval()
    for data in test_loader:
        data = data.to(DEVICE)
        pred_out = model(data.x, data.edge_index)
        acc = binary_acc(
                torch.sigmoid(pred_out), data.y.type(torch.FloatTensor)
                ) 
        epoch_test_acc += acc.item()

    print(
        f"""
        Epoch {e+0:03}: 
        | Loss: {epoch_loss/len(train_loader):.5f}
        | Train Acc: {epoch_train_acc/len(train_loader):.3f}
        | Test Acc : {epoch_test_acc/len(test_loader):.3f}"""
    )
