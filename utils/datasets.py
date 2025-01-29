import os

import numpy as np
import pandas as pd
import tifffile as tiff
import torch
from torch.utils.data import Dataset


class PDDataset(Dataset):
    def __init__(self, csv_file_path, img_dir, transform=None):
        self.csv_file_path = csv_file_path
        self.img_dir = img_dir

        self.df = pd.read_csv(csv_file_path, low_memory=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = str(self.df.iloc[idx]['FULL_FILE_PATH'])
        img_name = os.path.join(self.img_dir, os.path.basename(file_name).replace('nii.gz', 'tiff'))
        img = tiff.imread(img_name).astype('f4')

        if self.transform:
            img = self.transform(img)

        return np.expand_dims(img, 0), self.df.iloc[idx]['Group_numeric']


class UKBBT1Dataset(Dataset):

    def __init__(self, csv_file_path, img_dir, transform=None):
        self.csv_file_path = csv_file_path
        self.img_dir = img_dir

        self.df = pd.read_csv(csv_file_path, low_memory=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = str(int(self.df.iloc[idx]['eid'])) + '.tiff'
        img = tiff.imread(self.img_dir / img_name)

        if self.transform:
            img = self.transform(img)

        return self.df.iloc[idx]['Sex'], self.df.iloc[idx]['Age'], self.df.iloc[idx]['BMI'], img


class TorchDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path

    def __len__(self):
        return len(os.listdir(self.data_path))

    def __getitem__(self, index):
        file_path = os.path.join(self.data_path, str(index))
        return torch.load(file_path)
