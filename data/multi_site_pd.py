import os

import lightning as L
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class PDDataModule(L.LightningDataModule):
    def __init__(self, csv: str, img_dir: str, batch_size: int = 32):
        super().__init__()
        self.csv = csv
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

    def prepare_data(self):
        df = pd.read_csv(self.csv)
        df = df.rename(columns={"Study": "Site", "Scanner_type": "Type"})

        mask = ~df["Site"].str.contains("|".join(["SALD"]), case=True, na=False)
        df = df[mask]
        df.reset_index(drop=True, inplace=True)

        df, _ = categorize(df, "Type")
        df, _ = categorize(df, "Site")
        df = binarize(df, "Group", "PD")
        df = binarize(df, "Sex", "M")

        df["strat_column"] = (
            df["Site"].astype("str")
            + "_"
            + df["Sex"].astype("str")
            + "_"
            + df["Group"].astype("str")
        )

        strata_counts = df["strat_column"].value_counts()
        strata_to_drop = strata_counts[strata_counts < 3].index
        df = df[~df["strat_column"].isin(strata_to_drop)]

        train_val, test = train_test_split(
            df,
            test_size=0.2,
            stratify=df["strat_column"],
            random_state=42,
        )

        train, val = train_test_split(
            train_val,
            test_size=0.1 / 0.8,
            stratify=train_val["strat_column"],
            random_state=42,
        )

        train.to_csv("data/multi_site_pd_train.csv", index=False)
        test.to_csv("data/multi_site_pd_test.csv", index=False)
        val.to_csv("data/multi_site_pd_val.csv", index=False)

    def setup(self, stage: str):
        self.pd_train = PDDataset(
            "data/multi_site_pd_train.csv", self.img_dir, transform=self.transform
        )
        self.pd_val = PDDataset(
            "data/multi_site_pd_val.csv", self.img_dir, transform=self.transform
        )
        self.pd_test = PDDataset(
            "data/multi_site_pd_test.csv", self.img_dir, transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(self.pd_train, batch_size=self.batch_size, num_workers=15)

    def val_dataloader(self):
        return DataLoader(self.pd_val, batch_size=self.batch_size, num_workers=15)

    def test_dataloader(self):
        return DataLoader(self.pd_test, batch_size=self.batch_size, num_workers=15)


class PDDataset(Dataset):
    def __init__(self, csv, img_dir, transform=None):
        self.csv_file_path = csv
        self.img_dir = img_dir

        self.df = pd.read_csv(csv, low_memory=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = str(self.df.iloc[idx]["Subject"]) + ".nii.gz"
        img_name = os.path.join(self.img_dir, os.path.basename(file_name))
        img = nib.load(img_name).get_fdata().astype("f4")

        if self.transform:
            img = self.transform(img)

        PD = torch.tensor(self.df.iloc[idx]["Group"])
        sex = torch.tensor(self.df.iloc[idx]["Sex"])
        age = torch.tensor(self.df.iloc[idx]["Age"])
        study = torch.tensor(self.df.iloc[idx]["Site"])
        scanner_type = torch.tensor(self.df.iloc[idx]["Type"])

        return torch.unsqueeze(img, 0), PD, sex, age, study, scanner_type


def categorize(df, col):
    type_unique = df[col].unique()
    type_categories = {k: v for k, v in zip(type_unique, np.arange(len(type_unique)))}
    df[col] = df[col].map(type_categories)
    return df, type_categories


def binarize(df, col, val):
    df[col] = (df[col] == val).astype("int")
    return df
