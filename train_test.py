import torch
from data_loader.sumo_incident_dataset import SumoIncidentDataset
from models.End_to_End.nets import FAE_FeatGraphConv
from typing import Final
from sklearn.metrics import mean_squared_error as scimse
from torch_geometric.utils import to_undirected
import numpy as np
from sklearn.model_selection import KFold
import copy
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from utils.functions import index_to_mask
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

#######################
# TODO:
# 1. Instantiate Model
# 2. Instantiate Data
LEARNING_RATE: Final[float] = .001
EPOCHS: Final[int] = 300  # orginally 300 ( should be 20000 apparantly )
SUFFIX: Final[str] = "third"

def run():
    #######################
    # Data
    #
    # Could perhaps try normalizing the features as done in the pytorch geometric tutorial
    data = SumoIncidentDataset('data', 'SUMO_incident')    
    data = data[0]

    #######################
    # model
    #
    model = FAE_FeatGraphConv
    supervised_prediction_eval(model, data)
    
# TODO: either move data instant into here, global, or make data parameter here more granular
def train_epoch(model, data, optimizer, 
                exp_num=None, criterion=None):
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(
            output[data.train_mask], 
            data.y[data.train_mask, exp_num]
                .reshape([-1, 1])
            )
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(model, data, exp_num, criterion):
    model.eval()
    output = model(data)
    loss = criterion(
            output[data.test_mask],
            data.y[data.test_mask, exp_num]
                .reshape([-1, 1])
            )
    return loss.item()


# This kicks everything off
def supervised_prediction_eval(model_class, data):
    loss_train = []
    criterion = torch.nn.MSELoss()
    kf = KFold(n_splits=3)
    kf_feats = KFold(n_splits=3)
    mse = []

    #######################
    # Tensorboard
    #
    writer = SummaryWriter('runs/FAE_FeatGraphConv_' + SUFFIX)

    for k, train_test_indices in enumerate(kf.split(data.x)):
        print('Fold number: {:d}'.format(k))
        y_pred = []
        train_index, test_index = train_test_indices

        eval_data = copy.deepcopy(data)
        print("Y dimension: ",data.y.shape)
        print("y size", data.y.size(0))

        # This creates 2 arrays of indices, evenly split
        train_feats_indeces, test_feats_indeces = next(
                kf_feats.split(np.arange(data.y.size(0)))
                )

        print("train_feats_indeces", train_feats_indeces)
        print("test_feats_indeces", test_feats_indeces)

        eval_data.x = data.x
        eval_data.y = data.x
        eval_data.train_mask = index_to_mask(train_index, eval_data.x.size(0))
        eval_data.test_mask = index_to_mask(test_index, eval_data.x.size(0))

        # ??????????????????????????????????????
        torch.manual_seed(12345)             # ? 
        if torch.cuda.is_available():        # ?
            torch.cuda.manual_seed_all(12345)# ?
        # ??????????????????????????????????????

        # exp_num is actually the number of columns because a seperate model is trained for each column according 
        # to the paper.
        for exp_num in range(eval_data.y.size(1)):
            model = model_class(eval_data.num_features).to('cpu')
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            best_loss = 1e9

            for epoch in range(1, EPOCHS + 1):
                loss_train = train_epoch(model, eval_data, optimizer, exp_num, criterion)
                if loss_train < best_loss:
                    best_loss = loss_train
                    best_model = copy.deepcopy(model)
                # Per each epoch
                print(
                    f"""
                    Epoch {epoch}:
                    | Exp_num   : {exp_num}
                    | Loss: {loss_train:.3f}
                    """
                )
                writer.add_scalar(f"Train Loss; Feature: {exp_num}, Fold: {k}",
                        loss_train, epoch)
            loss_test = test(best_model, eval_data, exp_num, criterion)
            # Per exp_num print out
            print(
                f"""
                Feature - {exp_num}:
                | Best Loss : {best_loss:.3f}
                | Test Loss : {loss_test:.3f}"""
            )
            writer.add_scalar(f"Final Train Loss; Feature: {exp_num}",
                    best_loss, k)

            writer.add_scalar(f"Test Loss; Feature: {exp_num}",
                    loss_test, k)
                  
            with torch.no_grad():
                y_pred.append(best_model(eval_data))

            ##############################################
            # Save models for each feature
            #
            torch.save(best_model.state_dict(), f"finModels/feature{exp_num}_model.pth")
             

        for i in range(eval_data.y.size(1)):
            mse.append(scimse(y_pred[i][eval_data.test_mask.cpu().numpy()].cpu().numpy(),
                eval_data.y[eval_data.test_mask, i].cpu().numpy().reshape([-1, 1]))) # This may cause issues
    print('Average+-std Error for test expression values: {:.5f}+-{:.5f}'.format(np.mean(mse), np.std(mse)))
    return mse

if __name__ == "__main__":
    run()
