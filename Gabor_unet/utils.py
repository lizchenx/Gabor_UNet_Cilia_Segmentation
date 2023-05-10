#adapted from the paper: https://par.nsf.gov/servlets/purl/10354670 

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
from keras import backend as K

def process_mask(img): #make mask into 0 or 255
        imm = np.zeros(img.shape)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i,j] > 127:
                    imm[i,j] = 255
                else:
                    imm[i,j] =0
        
        return imm
    
#gabor filter 44
def gabor_filter(image):
    ksize=9
    sigma = 1
    theta = 3/4 * np.pi 
    lamda = 3/4 * np.pi
    gamma = 0.25
    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
    
    image = cv2.filter2D(image,cv2.CV_8UC1, kernel)
    return image

#evaluation metrics
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def accuracy(pred, target):
    return np.sum(np.equal(pred, target)) / (128**2)

def dice(pred, target):
    return 2 * np.logical_and(pred, target).sum() / (pred.sum() + target.sum() + K.epsilon())

def iou(pred, target):
    inter = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum() + K.epsilon()
    return inter / union

