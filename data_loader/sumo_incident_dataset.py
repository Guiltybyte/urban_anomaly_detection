import torch
from torch_geometric.data import InMemoryDataset, download_url 
import data_loader.load_csv
import os.path
import numpy as np
import pickle

class SumoIncidentDataset(InMemoryDataset):
    def __init__(
            self,
            root: str,
            name: str,
            transform=None,
            pre_transform=None,
            pre_filter=None
        ):
        self.name = name
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
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

    def process(self):
        # Read data into huge data list
        data_list = data_loader.load_csv.get_data_list(
                os.path.join(self.root, self.name)
                )

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
