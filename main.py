import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.loader import DataLoader
from typing import Final
import data_loader.load_csv
import models.gcn
from data_loader.sumo_incident_dataset import SumoIncidentDataset
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


# Hyper-Parameters
EPOCHS: Final[int] = 25 # 50
BATCH_SIZE: Final[int] = 18 # 32 
LEARN_RATE: Final[float] = 0.001

# Setup and test & train dataset
data_set = SumoIncidentDataset('data', 'SUMO_incident')
data_set = data_set.shuffle()
print("Length Dataset: ", len(data_set))
print("Length Dataset: ", (len(data_set) // 10)*6)
print("Length test set: ", ((len(data_set) // 10)*8)-((len(data_set) // 10)*6))
print("Length validate set: ", len(data_set) - ((len(data_set) // 10)*8))
train_dataset = data_set[:(len(data_set) // 10)*6] # 60% train
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset  = data_set[(len(data_set) // 10)*6:(len(data_set) // 10)*8] # 20% test 
test_loader   = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
validate_dataset  = data_set[(len(data_set) // 10)*8:] # 20% validate
validate_loader   = DataLoader(validate_dataset, batch_size=BATCH_SIZE, shuffle=True)
# test_dataset  = data_set[:len(data_set) // 20]
# test_loader   = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
# validate_dataset  = data_set[len(data_set) // 20: len(data_set) // 10]
# validate_loader   = DataLoader(validate_dataset, batch_size=len(validate_dataset), shuffle=True)


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
trainLossAtEachEpoch = []
testLossAtEachEpoch = []
validateLossAtEachEpoch = []
trainAccAtEachEpoch = []
testAccAtEachEpoch = []
validateAccAtEachEpoch = []
for e in range(EPOCHS):
    epoch_loss = 0 
    epoch_train_acc = 0
    epoch_test_loss = 0
    epoch_test_acc = 0
    epoch_val_acc = 0
    epoch_val_loss = 0

    model.train()
    for data in train_loader:
        data = data.to(DEVICE)
        optimizer.zero_grad()
        pred_out = model(data.x, data.edge_index)
        # Cross Entropy
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(
                pred_out, 
                data.y.type(torch.FloatTensor)
                )

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
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(
                pred_out, 
                data.y.type(torch.FloatTensor)
                )
        acc = binary_acc(
                torch.sigmoid(pred_out), data.y.type(torch.FloatTensor)
                ) 
        epoch_test_loss += loss.item()
        epoch_test_acc += acc.item()


    print(
        f"""
        Epoch {e+0:03}: 
        | Loss: {epoch_loss/len(train_loader):.5f}
        | Train Acc: {epoch_train_acc/len(train_loader):.3f}
        | Test Acc : {epoch_test_acc/len(test_loader):.3f}"""
    )
    trainLossAtEachEpoch.append(epoch_loss/len(train_loader))
    trainAccAtEachEpoch.append(epoch_train_acc/len(train_loader))
    testAccAtEachEpoch.append(epoch_test_acc/len(test_loader))
    testLossAtEachEpoch.append(epoch_test_loss/len(test_loader))

    fpr, tpr, _ = roc_curve([1, 0], [0, 1])
    for data in validate_loader:
        data = data.to(DEVICE)
        pred_out = model(data.x, data.edge_index)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(
                pred_out, 
                data.y.type(torch.FloatTensor)
                )
        acc = binary_acc(
                torch.sigmoid(pred_out), data.y.type(torch.FloatTensor)
                ) 
        epoch_val_acc += loss.item() 
        epoch_val_loss += acc.item() 
        fpr, tpr, _ = roc_curve(data.y.numpy(), torch.sigmoid(pred_out).detach().numpy())
        nopred_f, nopred_t, _ = roc_curve(data.y.numpy(), np.zeros(data.y.numpy().shape))
    
    validateAccAtEachEpoch.append(epoch_val_acc/len(validate_loader))
    validateLossAtEachEpoch.append(epoch_val_loss/len(validate_loader))

# ROC curve
# plot1 = plt.figure(1)
# plt.axis([0, 1, 0, 1.05])
# plt.grid()
# plt.plot(nopred_f, nopred_t, 'k--')
# plt.plot(fpr, tpr)
# plt.title("ROC Curve")
# plt.ylabel("True Positive rate")
# plt.xlabel("False Positive rate")

# learning convergence 
plot2 = plt.figure(2)
plt.plot(range(0, EPOCHS), trainLossAtEachEpoch)
plt.title("train Optimization Convergence")
plt.ylabel("Loss")
plt.xlabel("Epoch")

# Accuracy 
# plot3 = plt.figure(3)
# plt.plot(range(0, EPOCHS), trainAccAtEachEpoch)
# plt.title("train Accuracy")
# plt.ylabel("Accuracy (%)")
# plt.xlabel("Epoch")

# test and validate accuracy
plot4 = plt.figure(4)
plt.plot(range(0, EPOCHS), testAccAtEachEpoch, label="test")
plt.plot(range(0, EPOCHS), validateAccAtEachEpoch, label="validate")
plt.title("test & Validate Accuracy")
plt.legend()
plt.ylabel("Accuracy (%)")
plt.xlabel("Epoch")

# test and validate loss
plot5 = plt.figure(5)
plt.plot(range(0, EPOCHS), testLossAtEachEpoch, label="test")
plt.plot(range(0, EPOCHS), validateLossAtEachEpoch, label="validate")
plt.title("test & validate Loss")
plt.legend()
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.show()
