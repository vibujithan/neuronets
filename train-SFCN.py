import gc
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.sfcn import SFCN
import pytorch_lightning as pl


from utils.datasets import TorchDataset as TD


class LitSFCN(pl.LightningModule):
    def __init__(self, channel_number=[28, 58, 128, 256, 256, 64], output_dim=1):
        super().__init__()
        self.model = SFCN(output_dim=output_dim, channel_number=channel_number)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y = torch.squeeze(y)
        y_pred = self.model(x).squeeze()
        loss = self.criterion(y_pred, y.to(torch.float))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y = torch.squeeze(y)
        y_pred = self.model(x).squeeze()
        loss = self.criterion(y_pred, y.to(torch.float))
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-5, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


def main():
    train_path = "data/train"
    val_path = "data/val"

    batch_size = 8

    train_loader = DataLoader(
        TD(train_path), batch_size=batch_size, shuffle=True, num_workers=16
    )
    val_loader = DataLoader(TD(val_path), batch_size=batch_size, num_workers=16)

    model = LitSFCN(output_dim=1, channel_number=[28, 58, 128, 256, 256, 64])

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=100,
        precision=16,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
