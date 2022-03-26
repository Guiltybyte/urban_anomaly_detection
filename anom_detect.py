import torch
from data_loader.sumo_incident_dataset import SumoIncidentDataset
from models.End_to_End.nets import FAE_FeatGraphConv
import numpy as np
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

@torch.no_grad()
def test(model, data, exp_num, criterion): 
    model.eval()
    output = model(data)
    loss = criterion(
            output,
            data.x[:, exp_num]
                .reshape([-1, 1])
            )
    return loss.item()

@torch.no_grad()
def predicts(model, data, exp_num):
    model.eval()
    print("data.x.shape", data.x.shape)
    output = model(data)
    return (output - data.x[:, exp_num].reshape([-1, 1])) ** 2

#######################
# Dataset
#

data = SumoIncidentDataset('data', 'SUMO_incident')    
data = data[0]
print(data)

#######################
# Load Models
#

# 0
num_vehicles_model = FAE_FeatGraphConv(data.num_features)
num_vehicles_model.load_state_dict(torch.load('finModels/feature0_model.pth'))

# 1
flow_model = FAE_FeatGraphConv(data.num_features) 
flow_model.load_state_dict(torch.load('finModels/feature1_model.pth'))

# 2
occupancy_model = FAE_FeatGraphConv(data.num_features)
occupancy_model.load_state_dict(torch.load('finModels/feature2_model.pth'))

# 3
meanSpeed_model = FAE_FeatGraphConv(data.num_features)
meanSpeed_model.load_state_dict(torch.load('finModels/feature3_model.pth'))

# 4
lastDetection_model = FAE_FeatGraphConv(data.num_features)
lastDetection_model.load_state_dict(torch.load('finModels/feature4_model.pth'))

###################################################################
# 1. Determine Threshold by getting the MSE of each model 
#  Note: should probably be using a different dataset
criterion = torch.nn.MSELoss()


def predict():
    """
    Predict Anomaly (y label)  based on threshold
    """

    # num_vehicles_loss  : 1.2758539915084839 
    # flow_loss          : 10.219118118286133
    # occupancy_loss     : 0.40761440992355347
    # meanSpeed_loss     : 0.25336578488349915
    # lastDetection_loss : 2.463820219039917
    num_vehicles_loss = test(num_vehicles_model, data, 0, criterion) 
    flow_loss = test(flow_model, data, 1, criterion)
    occupancy_loss = test(occupancy_model, data, 2, criterion)
    meanSpeed_loss = test(meanSpeed_model, data, 3, criterion)
    lastDetection_loss = test(lastDetection_model, data, 4, criterion)
    num_vehicles_loss = 1.2758539915084839 
    flow_loss = 10.219118118286133
    occupancy_loss = 0.40761440992355347
    meanSpeed_loss = 0.25336578488349915
    lastDetection_loss = 2.463820219039917
    yeep = np.array(
            [num_vehicles_loss, flow_loss, occupancy_loss, meanSpeed_loss, lastDetection_loss]
            )
    threshold = yeep.mean()
    print("Threshold: ", threshold)

    # Make prediction for single slice of each data set 
    print("Starting predictions")
    nv_losses = predicts(num_vehicles_model, data, 0).cpu().detach().numpy() 
    nv_threshold = np.mean(nv_losses) + np.std(nv_losses)
    print("nv_threshold: ", nv_threshold)
    f_losses = predicts(flow_model, data, 1).cpu().detach().numpy() 
    f_threshold = np.mean(f_losses) + np.std(f_losses)
    print("f_threshold", f_threshold)
    o_losses = predicts(occupancy_model, data, 2).cpu().detach().numpy() 
    o_threshold = np.mean(o_losses) + np.std(o_losses)
    print("o_threshold", o_threshold)
    ms_losses = predicts(meanSpeed_model, data, 3).cpu().detach().numpy() 
    ms_threshold = np.mean(ms_losses) + np.std(ms_losses)
    print("ms_threshold", ms_threshold)
    ld_losses = predicts(lastDetection_model, data, 4).cpu().detach().numpy() 
    ld_threshold = np.mean(ld_losses) + np.std(ld_losses)
    print("ld_threshold", ld_threshold)
    mean_losses = np.mean([nv_losses, f_losses, o_losses, ms_losses, ld_losses], axis=0)
    # print("Mean Losses Shape: ", mean_losses.shape)

    #######################
    # Get Anomalous indexes 
    #
    y_actual = data.y.cpu().detach().numpy()
    anomaly_idxs = np.where(y_actual == 1)[0]
    print(anomaly_idxs.shape)
    print(mean_losses[anomaly_idxs])
    print(mean_losses[anomaly_idxs])
    non_anomaly_mask = np.ones(mean_losses.size, dtype=bool)
    non_anomaly_mask[anomaly_idxs] = False
    non_anomaly_losses = mean_losses[non_anomaly_mask]
    print(non_anomaly_losses)
    print("Shape non_anomaly_losses: ", non_anomaly_losses.shape)
    print("Shape non_anomaly_losses: ", non_anomaly_losses.flatten().shape)
    # plt.title("Visualize Squared Errors")
    # plt.scatter(range(len(non_anomaly_losses[:,0])), non_anomaly_losses[:,0], color='blue', s=2,marker=',') 
    # plt.scatter(range(len(mean_losses[anomaly_idxs][:,0])), mean_losses[anomaly_idxs][:,0], color='red', s=2,marker=',')
    # plt.legend(["Normal", "Anomalous"])
    # plt.ylabel("Mean Squared error")
    # plt.show()
    plt.title("Squared Errors")
    kwargs = dict(alpha=0.5, bins=100, density=True)
    plt.hist(mean_losses[anomaly_idxs][:,0], **kwargs, color='r', label='Anomalous')
    plt.hist(non_anomaly_losses[:,0], **kwargs, color='b', label='normal')
    plt.legend()
    plt.show()
    ########################
    # Thresholding operation
    # Threshold per each model output
    np_idx = np.where(nv_losses > nv_threshold)[0] 
    f_idx  = np.where(f_losses > f_threshold)[0]
    o_idx  = np.where(o_losses > o_threshold)[0]
    ms_idx = np.where(ms_losses > ms_threshold)[0]
    ld_idx = np.where(ld_losses > ld_threshold)[0]
    print(np_idx)
    print(f_idx )
    print(o_idx )
    print(ms_idx)
    print(ld_idx)
    print(anomaly_idxs)
predict()
