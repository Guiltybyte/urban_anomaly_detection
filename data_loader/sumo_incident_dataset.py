import torch
from torch_geometric.data import InMemoryDataset, download_url 
import data_loader.load_csv
import os.path
import numpy as np
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
import pickle

class SumoIncidentDataset(object):
    def __init__(
            self,
            root: str,
            name: str,
            num_timesteps_in: int,
            transform=None,
            pre_transform=None,
            pre_filter=None
        ):
        self.name = name
        self.root = root
        self.num_timesteps_in = num_timesteps_in
    
    @property
    def num_classes(self) -> int:
        return 2

    @property
    def num_node_features(self) -> int:
        """
        Returns Number of node features
        """
        return 5 

    @property
    def num_nodes(self) -> int:
        """
        Returns Number of node features
        """
        # TODO: Figure out how to avoid hard coding this
        return 104 

    @property
    def num_timesteps(self) -> int:
        """
        Returns number of future timesteps included in graph feature matrix
        """
        return self.num_timesteps_in

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.name, "processed")


    @property
    def raw_file_names(self) -> list:
        return ['edge_index.csv', 'nodes.csv', 'traffic_data.csv']

    @property
    def processed_file_names(self) -> list:
        return ['sumo_incident_dataset.pt']


    # def download(self):
        ## 1st download to self.raw_dir like below:
        # download_url(url, self.raw_dir)
        ## then unzip

    def get_data(self):
        dir_path = os.path.join(self.root, self.name, "processed")
        print("dir_path")
        if os.path.isfile(os.path.join(dir_path, 'x.npy')) and os.path.isfile(os.path.join(dir_path, 'y.npy')) and os.path.isfile(os.path.join(dir_path, 'edge_index')):
                # load the files
            x = np.load(os.path.join(dir_path, 'x.npy'))
            y = np.load(os.path.join(dir_path, 'y.npy'))
            with open(os.path.join(dir_path, 'edge_index'), 'rb') as f:
                edge_index = pickle.load(f)

        else:    
            edge_index, x, y = data_loader.load_csv.get_data_list(
                os.path.join(self.root, self.name), self.num_timesteps_in)
            # save files
            np.save(os.path.join(dir_path, 'x.npy'), x)
            np.save(os.path.join(dir_path, 'y.npy'), y)
            with open(os.path.join(dir_path, 'edge_index'), 'wb') as f:
                pickle.dump(edge_index, f)

        return StaticGraphTemporalSignal(
                edge_index, None, x, y
                )
