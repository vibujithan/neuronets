import argparse
import os
import random

import numpy as np
import tifffile as tiff
import torch
from torch.utils.data import DataLoader, random_split


def split_train_val(train_dataset, train_split=0.8, batch_size=None):
    # Calculate split sizes
    m = len(train_dataset)
    train_size = int(m * train_split)
    val_size = m - train_size

    # Split dataset
    train_data, val_data = random_split(train_dataset, [train_size, val_size],
                                        generator=torch.Generator().manual_seed(42))

    return DataLoader(train_data, batch_size=batch_size), DataLoader(val_data, batch_size=batch_size)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def compute_mean_slice(directory):
    images = []

    for filename in os.listdir(directory):
        if filename.endswith('.tiff') or filename.endswith('.tif'):
            img_path = os.path.join(directory, filename)
            img = tiff.imread(img_path)
            img = img.astype('f8')
            maxv = np.max(img)
            minv = np.min(img)
            img = ((img - minv) / maxv).astype('f4')
            images.append(img)
    if not images:
        raise ValueError("No TIFF images found in the directory")

    images_stack = np.stack(images)
    mean_image = np.mean(images_stack, axis=0)

    return mean_image


def seed_all(seed, deterministic=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    g = torch.Generator()
    g.manual_seed(seed)
    return g
