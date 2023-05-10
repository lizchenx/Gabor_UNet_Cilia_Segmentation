#adapted from the paper https://par.nsf.gov/servlets/purl/10354670

import torch
import pytorch_lightning as pl
from dataset import CiliaData
import config
from model import UNet
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from callbacks import MyPrintingCallback, EarlyStopping
from pytorch_lightning.strategies import DeepSpeedStrategy


logger = TensorBoardLogger('tb_logs', name = 'G44')
log_rate=10
input_channels = 2

model = UNet(input_channels=input_channels,
        log_rate = log_rate,
        num_classes=config.NUM_CLASSES,
        )

#data
dm = CiliaData(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,       
    )
dm._setup()

lr_monitor = LearningRateMonitor()
trainer = pl.Trainer(
    logger = logger,
    accelerator=config.ACCELERATOR,
    devices=config.DEVICES,
    max_epochs=config.NUM_EPOCHS,
    callbacks=[lr_monitor], 
    log_every_n_steps = 10,
    strategy = 'ddp',
    enable_model_summary=True,
    )

trainer.fit(model, dm)
trainer.validate(model, dm)
trainer.test(model, dm)
