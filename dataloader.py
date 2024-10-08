from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch

class CCV(Dataset):
    def __init__(self, path):
        self.data1 = np.load(path+'STIP.npy').astype(np.float32)
        scaler = MinMaxScaler()
        self.data1 = scaler.fit_transform(self.data1)
        self.data2 = np.load(path+'SIFT.npy').astype(np.float32)
        self.data3 = np.load(path+'MFCC.npy').astype(np.float32)
        self.labels = np.load(path+'label.npy')
    def __len__(self):
        return 6773
    def __getitem__(self, idx):
        x1 = self.data1[idx]
        x2 = self.data2[idx]
        x3 = self.data3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], \
            torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


class Hdigit():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'Hdigit.mat')
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(10000,)
        self.V1 = data['data'][0][0].T.astype(np.float32)
        self.V2 = data['data'][0][1].T.astype(np.float32)
    def __len__(self):
        return 10000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class cifar_100():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'cifar100.mat')
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(50000,)
        self.V1 = data['data'][0][0].T.astype(np.float32)
        self.V2 = data['data'][1][0].T.astype(np.float32)
        self.V3 = data['data'][2][0].T.astype(np.float32)
    def __len__(self):
        return 50000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class YoutubeFace_sel_fea():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'YoutubeFace_sel_fea.mat')
        self.Y = data['Y'].astype(np.int32).reshape(101499,)
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)
        self.V3 = data['X'][2][0].astype(np.float32)
        self.V4 = data['X'][3][0].astype(np.float32)
        self.V5 = data['X'][4][0].astype(np.float32)
    def __len__(self):
        return 101499
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        x4 = self.V4[idx]
        x5 = self.V5[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3), torch.from_numpy(x4), torch.from_numpy(x5)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class synthetic3d():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'synthetic3d.mat')
        self.Y = data['Y'].astype(np.int32).reshape(600,)
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)
        self.V3 = data['X'][2][0].astype(np.float32)
    def __len__(self):
        return 600
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

def load_data(dataset):
    if dataset == "CCV":
        dataset = CCV('./data/')
        dims = [5000, 5000, 4000]
        view = 3
        data_size = 6773
        class_num = 20
    elif dataset == "Hdigit":
        dataset = Hdigit('./data/')
        dims = [784, 256]
        view = 2
        data_size = 10000
        class_num = 10
    elif dataset == "Cifar100":
        dataset = cifar_100('./data/')
        dims = [512, 2048, 1024]
        view = 3
        data_size = 50000
        class_num = 100
    elif dataset == "YouTubeFace":
        dataset = YoutubeFace_sel_fea('./data/')
        dims = [64, 512, 64, 647, 838]
        view = 5
        data_size = 101499
        class_num = 31
    elif dataset == 'Synthetic3d':
        dataset = synthetic3d('./data/')
        dims = [3, 3, 3]
        view = 3
        data_size = 600
        class_num = 3
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num
