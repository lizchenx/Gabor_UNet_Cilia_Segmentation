#adapted from the study https://par.nsf.gov/servlets/purl/10354670

from dataset import CiliaData
from model import UNet
import pytorch_lightning as pl
from utils import recall_m, precision_m, iou, accuracy, dice
import config
import numpy as np
import math
import os
from glob import glob
from PIL import Image
from torchmetrics.classification import BinaryAccuracy



input_channels = 2
log_rate=10
num_classes=config.NUM_CLASSES

log_dir = 'tb_logs/G44'
dir_name = 'version_0'

dm = CiliaData(
               batch_size=1, 
        num_workers=config.NUM_WORKERS,
        )
dm._setup()

ckpt_path = glob(log_dir + "/" + dir_name+ "/checkpoints/*")
assert len(ckpt_path) == 1, f"Checkpoint directory has more than one {dir_name}"

model = UNet.load_from_checkpoint(ckpt_path[0], input_channels = input_channels,
log_rate=log_rate,
num_classes=config.NUM_CLASSES)
model.eval()
types=['orig']

os.mkdir('imgs/test0506/G44/' + dir_name)
os.mkdir('imgs/test0506/G44/' + dir_name + '/test/')


scores = [0, 0, 0, 0, 0]
count = 0

for x in dm.test_dataloader():
        img = x['image']       
        mask = x['mask']        
        pred = model(img).round()
        
        count += 1
        
        scores[0] += iou(mask[0][0].numpy().reshape(-1), pred[0][0].detach().numpy().reshape(-1))
        scores[1] += accuracy(mask[0][0].numpy().reshape(-1), pred[0][0].detach().numpy().reshape(-1))
        scores[2] += recall_m(mask[0][0].numpy().reshape(-1), pred[0][0].detach().numpy().reshape(-1))
        scores[3] += dice(mask[0][0].numpy().reshape(-1), pred[0][0].detach().numpy().reshape(-1))
        scores[4] +=precision_m(mask[0][0].numpy().reshape(-1), pred[0][0].detach().numpy().reshape(-1))
                     
        os.mkdir('imgs/test0506/G44/' + dir_name + '/test/' + x['id'][0].split('.')[0] + '/')
        for i, type in enumerate(types):
            image = (img[0][i] * 255).numpy().astype(np.uint8)
            Image.fromarray(image).save('imgs/test0506/G44/' + dir_name + '/test/' + x['id'][0].split('.')[0] + '/' + type + '.png')
        image = (mask[0][0] * 255).numpy().astype(np.uint8)
        Image.fromarray(image).save('imgs/test0506/G44/' + dir_name + '/test/' + x['id'][0].split('.')[0] + '/' + 'mask' + '.png')
        image = (pred[0][0] * 255).detach().numpy().astype(np.uint8)
        Image.fromarray(image).save('imgs/test0506/G44/' + dir_name + '/test/' + x['id'][0].split('.')[0] + '/' + 'pred' + '.png')

print(dir_name)
print("IOU = %f" % (scores[0] / count))
print("ACCURACY = %f" % (scores[1] / count))
print("DICE = %f" % (scores[3] / count))
print("RECALL = %f" % (scores[2] / count))
print("PRECISION = %f" % (scores[4] / count))

