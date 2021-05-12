import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from Hyper_parameters import HyperParams


class GTZANDataset(Dataset):
    '''
    Custom torch dataloader
    '''

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]


def get_label(filename):
    '''
    filename example: classical.00000.wav
    '''
    genre = filename.split(".")[0]
    label = HyperParams.genres.index(genre)
    return label


def load_dataset(name):
    x, y = [], []
    path = os.path.join(HyperParams.feature_path, name)
    for root, _, files in os.walk(path):
        for file in files:
            data = np.load(os.path.join(root, file))
            label = get_label(file)
            x.append(data)
            y.append(label)
    return np.stack(x), np.stack(y)


def myDataLoader():
    x_train, y_train = load_dataset("train")
    x_valid, y_valid = load_dataset("valid")
    x_test, y_test = load_dataset("test")

    # normalize
    mean = np.mean(x_train)
    std = np.std(x_train)
    x_train = (x_train-mean)/std
    x_valid = (x_valid-mean)/std
    x_test = (x_test-mean)/std

    train = GTZANDataset(x_train, y_train)
    valid = GTZANDataset(x_valid, y_valid)
    test = GTZANDataset(x_test, y_test)

    train_loader = DataLoader(
        train, batch_size=HyperParams.batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(
        valid, batch_size=HyperParams.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(
        test, batch_size=HyperParams.batch_size, shuffle=False, drop_last=False)

    return train_loader, valid_loader, test_loader
