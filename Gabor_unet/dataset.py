
# adapted from https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/pytorch_lightning/10.%20Multi-GPU/dataset.py 
#reference: Zain, Meekail, et al. "Low Level Feature Extraction for Cilia Segmentation." Proceedings of the Python in Science Conference. 2022.


from torch.utils.data import Dataset
import cv2

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
from os import listdir
from os.path import splitext
from glob import glob
from PIL import Image
import numpy as np
import torch
from utils import process_mask, gabor_filter
from keras import backend as K
from keras.utils import normalize

class CiliaData(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root = '../cilia_dataset/cilia326/patches/images5p82/%s/'
        self.mask_root = '../cilia_dataset/cilia326/patches/masks5p82/%s/' 
        
    
    def prepare_data(self):
        pass
    
    def _setup(self,val_percent=0.2):
        dataset = CiliaDataset(self.root%'train', self.mask_root%'train')
        n_val = int(len(dataset) * val_percent)
        n_train = len(dataset) - n_val
        self.train_data, self.val_data = random_split(dataset, [n_train, n_val])                    
        self.test_data = CiliaDataset(self.root % 'test', self.mask_root %'test')
        
    
    def train_dataloader(self):
        return DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

class CiliaDataset(Dataset):
    def __init__(self, imagePaths, maskPaths): 

        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.ids = [splitext(file)[0] for file in listdir(imagePaths) 
                    if not file.startswith(".")]
        
    def __len__(self):
        return len(self.ids)
    
    @classmethod
    def preprocess(self, img):
        img = img.astype(np.float32)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)
        img_min = img.min(axis=(1,2)).reshape(-1,1,1)
        img_max = img.max(axis=(1,2)).reshape(-1,1,1)
        img -= img_min
        
        img /= img_max - img_min #+ K.epsilon()

        img = torch.from_numpy(img)

        return img
  
    def preprocess2(self, img):
        img = img.astype(np.float32)/255
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(img)

        return img
    
    def __getitem__(self, i):

        idx = self.ids[i]
        mask_file = glob(self.maskPaths + idx+'.*')
        img_file = glob(self.imagePaths + idx+'.*')
         
        image = cv2.imread(img_file[0],0)
        
        #add Gabor filter
        fimg = gabor_filter(image)
        image = np.stack([image, fimg],axis = 0)
        
        mask = cv2.imread(mask_file[0], 0)
        mask = process_mask(mask)
        
        image = self.preprocess(image)
        mask = self.preprocess2(mask)

        
        return {"image": image, "mask": mask, "id": self.ids[i]}