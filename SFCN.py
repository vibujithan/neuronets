import sys

import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append("/data/Data/multi-site-PD")

from data.multi_site_pd import PDDataModule
from image_models.sfcn import SFCN
from torchmetrics.classification import BinaryAUROC

torch.set_float32_matmul_precision("medium")


class LitSFCN(L.LightningModule):
    def __init__(self, channel_number=[28, 58, 128, 256, 256, 64], output_dim=1):
        super().__init__()
        self.model = SFCN(output_dim=output_dim, channel_number=channel_number)
        self.criterion = nn.BCEWithLogitsLoss()
        self.auroc = BinaryAUROC(thresholds=None)

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

    def test_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y = torch.squeeze(y)
        y_pred = nn.functional.sigmoid(self.model(x).squeeze())
        auroc = self.auroc(y_pred, y)
        self.log("test_auroc", auroc)
        return auroc

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
    pd = PDDataModule(
        "/data/Data/multi-site-PD/data.csv",
        "/data/Data/multi-site-PD/images",
        batch_size=8,
    )

    model = LitSFCN(output_dim=1, channel_number=[28, 58, 128, 256, 256, 64])

    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=10,
        precision="bf16-mixed",
    )

    trainer.fit(model, pd)
    trainer.test(model, pd)


if __name__ == "__main__":
    main()
