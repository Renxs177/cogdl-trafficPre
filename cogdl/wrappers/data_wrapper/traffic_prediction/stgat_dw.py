from torch.utils.data import DataLoader, TensorDataset
from .. import DataWrapper
import numpy as np
import torch


class STGATDataWrapper(DataWrapper):
    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument("--batch_size", type=int, default=30)
        parser.add_argument("--n_his", type=int, default=12)
        parser.add_argument("--n_pred", type=int, default=9)
        parser.add_argument("--train_prop", type=int, default=0.03)
        parser.add_argument("--val_prop", type=int, default=0.02)
        parser.add_argument("--test_prop", type=int, default=0.01)
        parser.add_argument("--pred_length", type=int, default=288)
        # fmt: on

    def __init__(self, dataset, **args):
        super(STGATDataWrapper, self).__init__(dataset)
        self.dataset = dataset
        self.train_prop = args['train_prop']
        self.val_prop = args['val_prop']
        self.test_prop = args['test_prop']
        self.pred_length = args['pred_length']
        self.n_his = args['n_his']
        self.n_pred = args['n_pred']
        self.batch_size = args['batch_size']
        self.scaler = dataset.data.scaler


    def train_wrapper(self):
        train_data = self.dataLoad()[0]
        return DataLoader(train_data, self.batch_size, shuffle=True)

    def val_wrapper(self):
        val_data = self.dataLoad()[1]
        return DataLoader(val_data, self.batch_size, shuffle=False)

    def test_wrapper(self):
        # test_data = self.dataLoad()[2]
        # return DataLoader(test_data, self.batch_size, shuffle=False)

        pred_data = self.dataLoad()[3]
        return DataLoader(pred_data, self.pred_length + self.n_his + self.n_pred + 1, shuffle=False)


    def data_transform(self, data, n_his, n_pred, device):
        # data = slice of V matrix
        # n_his = number of historical speed observations to consider
        # n_pred = number of time steps in the future to predict
        num_nodes = data.shape[1]
        # T - 20 - 5 能观测的数据 num_obs是可以组成的训练预测对的数据量
        num_obs = len(data) - n_his - n_pred
        # 即 训练数据 为 num_obs 条，每一条 都是  n_his * num_nodes*1 大小的
        x = np.zeros([num_obs, n_his, num_nodes, 1])  # 347*20*50*1
        # 即 标签数据 为 num_obs 条， 每一条 是 1*50大小的
        y = np.zeros([num_obs, num_nodes])  # 347*50
        obs_idx = 0
        for i in range(num_obs):
            head = i
            tail = i + n_his
            # 有一个 n_his 大小的窗口 从 data 向下滑 取出每一个观测数据的 所有节点的信息
            x[obs_idx, :, :, :] = data[head: tail].reshape(n_his, num_nodes, 1)
            # 标签是只取了第五天的数据
            y[obs_idx] = data[tail + n_pred - 1]
            obs_idx += 1
        return torch.Tensor(x).to(device), torch.Tensor(y).to(device)


    def dataLoad(self):
        V = self.dataset.data.V
        len_train = round(self.dataset.data.num_samples * self.train_prop)
        len_val = round(self.dataset.data.num_samples * self.val_prop)  # 验证时间段 T*0.02
        len_test = round(self.dataset.data.num_samples * self.test_prop)
        train = V[: len_train]
        val = V[len_train: len_train + len_val]
        test = V[len_train + len_val: len_train + len_val + len_test]  # 测试时间段 T*0.01

        pred_set = V[len_train + len_val:len_train + len_val + self.pred_length + self.n_his + self.n_pred]


        train = np.nan_to_num(self.scaler.fit_transform(train))
        val = np.nan_to_num(self.scaler.transform(val))
        test = np.nan_to_num(self.scaler.transform(test))
        pred_set = np.nan_to_num(self.scaler.transform(pred_set))

        x_train, y_train = self.data_transform(train, self.n_his, self.n_pred, self.dataset.data.device)
        x_val, y_val = self.data_transform(val, self.n_his, self.n_pred, self.dataset.data.device)
        x_test, y_test = self.data_transform(test, self.n_his, self.n_pred, self.dataset.data.device)

        x_pred, y_pred = self.data_transform(pred_set, self.n_his, self.n_pred, self.dataset.data.device)

        # create torch data iterables for training
        train_data = TensorDataset(x_train, y_train)
        train_iter = DataLoader(train_data, self.batch_size, shuffle=True)
        val_data = TensorDataset(x_val, y_val)
        val_iter = DataLoader(val_data, self.batch_size, shuffle=False)
        test_data = TensorDataset(x_test, y_test)
        test_iter = DataLoader(test_data, self.batch_size, shuffle=False)

        pred_data = TensorDataset(x_pred, y_pred)
        pred_iter = DataLoader(pred_data, self.pred_length + self.n_his + self.n_pred + 1, shuffle=False)

        return [train_data, val_data, test_data, pred_data]


    def get_pre_timestamp(self):
        len_train = round(self.dataset.data.num_samples * self.train_prop)
        len_val = round(self.dataset.data.num_samples * self.val_prop)
        pred_set_timestamp = self.dataset.data.timestamp[len_train + len_val: len_train + len_val + self.pred_length][-self.pred_length:]
        return pred_set_timestamp














