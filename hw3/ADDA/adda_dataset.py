import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import glob
import cv2
import os

class AddaDataset(data.Dataset):
    def __init__(self, root, dataset, train):
        self.train = train
        if train:
            self.filename = sorted(glob.glob(os.path.join(root, dataset, 'train', '*png')))
            label = pd.read_csv(os.path.join(root, dataset, 'train.csv'))
            self.label = np.array(label['label'])
        else:
            self.filename = sorted(glob.glob(os.path.join(root, dataset, 'test', '*png')))
            label = pd.read_csv(os.path.join(root, dataset, 'test.csv'))
            self.label = np.array(label['label'])

    def __getitem__(self, index):
        image = cv2.imread(self.filename[index], 0)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transforms.ToTensor()(image).type(torch.float)
        label = int(self.label[index])
        return image, label
    
    def __len__(self):
        return len(self.filename)

class AddaTestset(data.Dataset):
    def __init__(self, root):
        self.filename = sorted(glob.glob(os.path.join(root, '*png')))
 
    def __getitem__(self, index):
        image = cv2.imread(self.filename[index], 0)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transforms.ToTensor()(image).type(torch.float)
        return image
    
    def __len__(self):
        return len(self.filename)