"""
Uses the raw .csv data files to create 
a series of Graphs (type: list[Data]) to 
train on.
"""
import torch 
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from typing import Final
import os

# python 3.9+ required for full dict type hint support
def create_mapping(node_raw_path: str) -> dict:
    """
    Maps each lengthy e1 detector ID string,
    to a unique integer ID.
    """
    # Optional arg specifies that 1st col should be treated as the index
    df = pd.read_csv(node_raw_path, index_col=0)
    mapping = {index: i for i, index in enumerate(df.index.unique())}
    return mapping

def load_edge_index(edge_raw_path: str, mapping: dict=None) -> torch.tensor:
    """
    Returns Graph Connectivity in COO format
    by reading file: @param edge_raw_path
    """
    COLUMNS: Final[list[str]] = ['source', 'target']
    df = pd.read_csv(edge_raw_path, usecols=COLUMNS)
    if mapping is not None:
        source: list[str] = apply_mapping(mapping, df.source.tolist())
        target: list[str] = apply_mapping(mapping, df.target.tolist())
    else:
        source: list[str] = df.source.tolist()
        target: list[str] = df.target.tolist()
    return torch.tensor([source, target], dtype=torch.long)

def apply_mapping(mapping: dict, lst: list) -> list:
    """
    Applies the given mapping to the given list
    and returns the new list.
    """
    # * means that the iterable will be unpacked into the list
    # passing map.get without args means we are passing the function itself
    return [*map(mapping.get, lst)]

def create_graph(traffic_data: pd.DataFrame, num_nodes: int, edge_index: torch.tensor) -> Data:
    """
    Creates a Single Unique graph, as better suits the dataset.
    """
    features = traffic_data.get(['num_vehicles', 'flow', 'occupancy', 'meanSpeed', 'lastDetectTime']).to_numpy()
    labels   = traffic_data.get(['label']).to_numpy().flatten()
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.float)
    graph = Data(x=x, y=y, edge_index=edge_index)
    return graph


def get_data_list(root_dir: str, verbose: bool=False):
    # Raw Data File Paths
    ROOT: Final[str] = os.path.join(os.path.dirname(__file__), '..', root_dir)
    NODE: Final[str] = os.path.join(ROOT, 'raw', 'nodes.csv')
    EDGE: Final[str] = os.path.join(ROOT, 'raw', 'edge_index.csv')
    DATA: Final[str] = os.path.join(ROOT, 'raw', 'traffic_data.csv')
    
    # 1. Create Node ID Mapping DONE 
    MAPPING: Final = create_mapping(NODE)
    NUM_NODES: Final[int] = len(MAPPING) 

    # 2. Load edge_index DONE
    EDGE_INDEX: Final[torch.tensor] = load_edge_index(EDGE, mapping=MAPPING)

    # Define columns to use read into dataFrame (Note: omitting time )
    COLUMNS: Final[list[str]] = [
            'loop', 'num_vehicles', 'flow', 'occupancy', 'meanSpeed','lastDetectTime', 'label' 
            ]
    df = pd.read_csv(DATA, usecols=COLUMNS)
    df.loop = df.loop.apply(lambda ID: MAPPING.get(ID)) # apply the ID mapping
    
    # 3.
    data_list = [create_graph(df, NUM_NODES, EDGE_INDEX)]


    if verbose:
        print("#------Finished DataSet!----------#")
        print("                 #NODES: ", NUM_NODES)
        print("                #GRAPHS: ", len(data_list))
        print("         GRAPH INSTANCE: ", data_list[0])
        print("             edge_index:\n", data_list[0].edge_index)
        print("eg. node feature matrix:\n", data_list[0].x)
        print("eg.        node labels :\n", data_list[0].y)

    return data_list 

if __name__ == "__main__":
    get_data_list("data/SUMO_incident", verbose=True)
