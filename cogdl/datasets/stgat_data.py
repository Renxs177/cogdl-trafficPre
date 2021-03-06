# Cora 数据集
import os.path as osp
import pickle as pkl
import sys
import pandas as pd
import numpy as np
import torch
from cogdl.data import Dataset, Graph
from cogdl.utils import remove_self_loops, download_url, untar, coalesce, MAE, CrossEntropyLoss
import numpy as np
import os
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
import torch
from datetime import datetime
import geopy.distance # to compute distances between stations
import glob
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import warnings
from numpy.core.umath_tests import inner1d

def files_exist(files):
    # 返回一个布尔变量
    return all([osp.exists(f) for f in files])


def raw_data_processByNumNodes(raw_dir, num_nodes):
    PeMS_daily = os.path.join(f'{raw_dir}', 'PeMS_20210501_20210630', '*')
    PeMS_metadata = os.path.join(f'{raw_dir}', 'PeMS_20210501_20210630', 'd07_text_meta.txt')
    output_dir = os.path.join(f'{raw_dir}')


    # Parameters
    outcome_var = 'avg_speed'
    files = glob.glob(PeMS_daily)
    files.remove(glob.glob(PeMS_metadata)[0])
    PeMS_columns = ['timestamp', 'station', 'district', 'freeway_num',
                    'direction_travel', 'lane_type', 'station_length',
                    'samples', 'perc_observed', 'total_flow', 'avg_occupancy',
                    'avg_speed']
    PeMS_lane_columns = lambda x: ['lane_N_samples_{}'.format(x),
                                   'lane_N_flow_{}'.format(x),
                                   'lane_N_avg_occ_{}'.format(x),
                                   'lane_N_avg_speed_{}'.format(x),
                                   'lane_N_observed_{}'.format(x)]
    PeMS_all_columns = PeMS_columns.copy()
    for i in range(1, 9):
        PeMS_all_columns += PeMS_lane_columns(i)
    # Randomly select stations to build the dataset
    np.random.seed(42)
    station_file = files[0]
    station_file_content = pd.read_csv(station_file, header=0, names=PeMS_all_columns)
    station_file_content = station_file_content[PeMS_columns]
    station_file_content = station_file_content.dropna(subset=[outcome_var])
    unique_stations = station_file_content['station'].unique()
    selected_stations = np.random.choice(unique_stations, size=num_nodes, replace=False)

    station_data = pd.DataFrame({col: []} for col in PeMS_columns)
    for station_file in tqdm(files):
        file_date_str = station_file.split(os.path.sep)[-1].split('.')[0]
        file_date = datetime(2021, int(file_date_str.split('_')[-2]),
                             int(file_date_str.split('_')[-1]))
        if file_date.weekday() < 5:
            # Read CSV
            station_file_content = pd.read_csv(
                station_file, header=0, names=PeMS_all_columns)
            # Keep only columns of interest
            station_file_content = station_file_content[PeMS_columns]
            # Keep stations
            station_file_content = station_file_content[
                station_file_content['station'].isin(selected_stations)]
            # Append to dataset
            station_data = pd.concat([station_data, station_file_content])
    # Drop the 11 rows with missing values
    station_data = station_data.dropna(subset=['timestamp', outcome_var])
    station_data.head()
    station_data.shape
    station_metadata = pd.read_table(PeMS_metadata)
    station_metadata = station_metadata[['ID', 'Latitude', 'Longitude']]
    # Filter for selected stations
    station_metadata = station_metadata[station_metadata['ID'].isin(selected_stations)]
    station_metadata.head()
    # Keep only the required columns (time interval, station ID and the outcome variable)
    station_data = station_data[['timestamp', 'station', outcome_var]]
    station_data[outcome_var] = pd.to_numeric(station_data[outcome_var])
    # Reshape the dataset and aggregate the traffic speeds in each time interval
    V = station_data.pivot_table(index=['timestamp'], columns=['station'], values=outcome_var, aggfunc='mean')
    V.head()
    V.shape
    # Compute distances
    distances = pd.crosstab(station_metadata.ID, station_metadata.ID, normalize=True)
    distances_std = []
    for station_i in selected_stations:
        for station_j in selected_stations:
            if station_i == station_j:
                distances.at[station_j, station_i] = 0
            else:
                # Compute distance between stations
                station_i_meta = station_metadata[station_metadata['ID'] == station_i]
                station_j_meta = station_metadata[station_metadata['ID'] == station_j]
                d_ij = geopy.distance.geodesic(
                    (station_i_meta['Latitude'].values[0], station_i_meta['Longitude'].values[0]),
                    (station_j_meta['Latitude'].values[0], station_j_meta['Longitude'].values[0])).m
                distances.at[station_j, station_i] = d_ij
                distances_std.append(d_ij)
    distances_std = np.std(distances_std)
    distances.head()
    W = pd.crosstab(station_metadata.ID, station_metadata.ID, normalize=True)
    epsilon = 0.5
    sigma = distances_std
    for station_i in selected_stations:
        for station_j in selected_stations:
            if station_i == station_j:
                W.at[station_j, station_i] = 0
            else:
                # Compute distance between stations
                d_ij = distances.loc[station_j, station_i]
                # Compute weight w_ij
                w_ij = np.exp(-d_ij ** 2 / sigma ** 2)
                if w_ij >= epsilon:
                    W.at[station_j, station_i] = 1
                else:
                    W.at[station_j, station_i] = 0
    W.head()
    # Save to file
    V = V.fillna(V.mean())
    V.to_csv(os.path.join(output_dir, 'V_{}.csv'.format(num_nodes)), index=True)
    W.to_csv(os.path.join(output_dir, 'W_{}.csv'.format(num_nodes)), index=False)
    station_metadata.to_csv(os.path.join(output_dir, 'station_meta_{}.csv'.format(num_nodes)), index=False)


def read_stgcn_data(folder, num_nodes):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    W = pd.read_csv(osp.join(folder, f"W_{num_nodes}.csv"))
    T_V = pd.read_csv(osp.join(folder, f"V_{num_nodes}.csv"))
    V = T_V.drop('timestamp',axis=1)
    num_samples, num_nodes = V.shape
    scaler = StandardScaler()


    # format graph for pyg layer inputs
    G = sp.coo_matrix(W)
    edge_index = torch.tensor(np.array([G.row, G.col]), dtype=torch.int64).to(device)
    edge_weight = torch.tensor(G.data).float().to(device)
    data = Graph()
    data.num_nodes = num_nodes
    data.num_samples = num_samples
    data.edge_index = edge_index
    data.edge_weight = edge_weight
    data.scaler = scaler
    data.V = V
    data.W = W
    data.timestamp = T_V['timestamp']
    data.node_ids = V.columns

    return data


class STGATDataset(Dataset):
    def __init__(self, root, name, num_stations):
        self.name = name
        self.url = f"https://cloud.tsinghua.edu.cn/f/5af7ea1a7d064c5ba6c8/?dl=1"
        self.num_stations = num_stations
        super(STGATDataset, self).__init__(root)
        print(self.processed_paths[0])
        self.data = torch.load(self.processed_paths[0])
        self.num_nodes = self.data.num_nodes

    @property
    def raw_file_names(self):
        names = [f"station_meta_{self.num_stations}.csv", f"V_{self.num_stations}.csv", f"W_{self.num_stations}.csv"]
        return names

    @property
    def processed_file_names(self):
        return ["data.pt"]


    def get(self, idx):
        assert idx == 0
        return self.data

    def download(self):
        if os.path.exists(self.raw_dir+r'\PeMS_20210501_20210630'):  # pragma: no cover
            return
        download_url(self.url, self.raw_dir, name=self.name + ".zip")
        untar(self.raw_dir, self.name + ".zip")

    def process(self):
        files = self.raw_paths
        if not files_exist(files):
            raw_data_processByNumNodes(self.raw_dir, self.num_stations)
        data = read_stgcn_data(self.raw_dir, self.num_stations)
        torch.save(data, self.processed_paths[0])


    def __repr__(self):
        return "{}".format(self.name)


    def get_evaluator(self):
        return MAE()

    def get_loss_fn(self):
        return torch.nn.MSELoss()


class PeMS_Dataset(STGATDataset):
    def __init__(self, data_path="data"):
        dataset = "pems-stgat"
        path = osp.join(data_path, dataset)
        super(PeMS_Dataset, self).__init__(path, dataset, num_stations=50)

