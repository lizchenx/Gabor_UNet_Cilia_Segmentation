#adapted from the paper https://par.nsf.gov/servlets/purl/10354670

import torch
from torch import optim
import torch.nn as nn
from torchvision.transforms.functional import pad
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import pytorch_lightning as pl
import numpy as np
import torchmetrics
from torchmetrics import Metric
from torcheval.metrics import BinaryAccuracy


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)

class Up(nn.Module):   
    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = False):
        super().__init__()
        self.upsample = None
        if bilinear:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=1),
            )
        else:
            self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)

        diff_h = x2.shape[2] - x1.shape[2]
        diff_w = x2.shape[3] - x1.shape[3]

        x1 = pad(x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class UNet(pl.LightningModule):
    def __init__(self, input_channels,log_rate, num_classes):
        super().__init__()
        
        self.num_classes=num_classes
        self.input_channels=input_channels
        features_start=64
        num_layers=5
        self.num_layers = num_layers
        self.log_rate = log_rate
        
        layers = [DoubleConv(input_channels, features_start)]

        feats = features_start
        for _ in range(num_layers - 1):
            layers.append(Down(feats, feats * 2))
            feats *= 2

        for _ in range(num_layers - 1):
            layers.append(Up(feats, feats // 2))
            feats //= 2

        layers.append(nn.Conv2d(feats, num_classes, kernel_size=1))

        self.layers = nn.ModuleList(layers)
        self.loss = nn.BCELoss()
        self.accuracy = BinaryAccuracy()

    def forward(self, x):
        xi = [self.layers[0](x)]

        for layer in self.layers[1 : self.num_layers]:
            xi.append(layer(xi[-1]))
        for i, layer in enumerate(self.layers[self.num_layers : -1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])
        return torch.sigmoid(self.layers[-1](xi[-1]))

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)

        scheduler = {
            "scheduler": CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2),
            "interval": "step",
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _generic_step(self, batch, batch_idx):
        true_masks = batch["mask"]
        images = batch["image"]
        masks = self.forward(images) 
        loss = self.loss(masks, true_masks)
        return loss, images, masks, true_masks

    def training_step(self, batch, batch_idx):
        loss = self._generic_step(batch, batch_idx)[0]
        self.log("train_loss", loss, rank_zero_only=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, images, masks, true_masks = self._generic_step(batch, batch_idx)
        self.log("val_loss", loss, rank_zero_only=True)
        if batch_idx==0 and self.current_epoch%self.log_rate==0 and hasattr(self.logger, 'experiment'):
            idx = np.random.randint(len(images))
            tensorboard = self.logger.experiment
            additional_channels=""
            logged_images = [images[idx][i:i+1] for i in range(images[idx].shape[0])]
            logged_images.extend([masks[idx], true_masks[idx]])
            logged_images = torch.cat(logged_images)
            
            tensorboard.add_images(f"Image/{additional_channels}Mask/Truth at epoch {self.current_epoch}", logged_images.unsqueeze(1))
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._generic_step(batch, batch_idx)[0]
        self.log("test_loss", loss, rank_zero_only=True)

        return loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("UNet")
        parser.add_argument(
            "--num_classes",
            type=int,
            default=1,
            help="number of classes for segmentation",
        )
        return parent_parser

