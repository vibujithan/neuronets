import os
from abc import ABC

import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner
from torchmetrics.classification import BinaryAUROC

from data.multi_site_pd import PDDataModule
from models.resnet import resnet34
from models.sfcn import SFCN
from models.sfcn_daft import SFCN_DAFT
from models.simple_vit import SimpleViT

torch.set_float32_matmul_precision("medium")
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class LitClassifier(L.LightningModule, ABC):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        self.auroc = BinaryAUROC(thresholds=None)
        self.learning_rate = 1e-5

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
        optimizer = optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


class LitSFCN(LitClassifier):
    def __init__(self, output_dim=1, channel_number=[28, 58, 128, 256, 256, 64]):
        super().__init__()
        self.model = SFCN(output_dim=output_dim, channel_number=channel_number)


class LitResNet(LitClassifier):
    def __init__(self, num_classes=1):
        super().__init__()
        self.model = resnet34(num_classes=num_classes)


class LitSFCNDAFT(LitClassifier):
    def __init__(self, output_dim=1, channel_number=[28, 58, 128, 256, 256, 64]):
        super().__init__()
        self.model = SFCN_DAFT(output_dim=output_dim, channel_number=channel_number)

    def training_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        tabluar_data = torch.stack(batch[2:5], dim=1).to(torch.float32)

        y_pred = self.model(x, tabluar_data).squeeze()
        loss = self.criterion(y_pred, y.to(torch.float))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        tabluar_data = torch.stack(batch[2:5], dim=1).to(torch.float32)

        y_pred = self.model(x, tabluar_data).squeeze()
        loss = self.criterion(y_pred, y.to(torch.float))
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        tabluar_data = torch.stack(batch[2:5], dim=1).to(torch.float32)

        y_pred = self.model(x, tabluar_data).squeeze()
        auroc = self.auroc(y_pred, y)
        self.log("test_auroc", auroc)
        return auroc


class LitSimpleViT(LitClassifier):
    def __init__(self):
        super().__init__()
        self.model = SimpleViT(
            image_size=(160, 192),
            image_patch_size=(16, 16),
            frames=160,
            frame_patch_size=8,
            num_classes=1,
            dim=1024,
            depth=6,
            heads=8,
            mlp_dim=512,
            channels=1,
            dim_head=64,
        )


def main():
    pd = PDDataModule(
        "/data/Data/multi-site-PD/data.csv",
        "/data/Data/multi-site-PD/images",
        batch_size=8,
    )

    model = LitSFCN()

    logger = TensorBoardLogger("lightning_logs", name=model.__class__.__name__)

    trainer = L.Trainer(
        fast_dev_run=False,
        accelerator="gpu",
        devices=1,
        max_epochs=200,
        precision="bf16-mixed",
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                min_delta=0.01,
                patience=5,
                verbose=False,
                mode="min",
            )
        ],
        logger=logger,
        log_every_n_steps=10,
    )

    tuner = Tuner(trainer)

    tuner.scale_batch_size(model, mode="power", datamodule=pd)
    tuner.lr_find(model, pd)

    trainer.fit(model, pd)
    trainer.test(model, pd)


if __name__ == "__main__":
    main()
