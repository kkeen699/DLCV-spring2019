import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import glob
import cv2
import os

class DsnDataset(data.Dataset):
    def __init__(self, root, dataset, train, channel):
        self.train = train
        self.channel = int(channel)
        if train:
            self.filename = sorted(glob.glob(os.path.join(root, dataset, 'train', '*png')))
            label = pd.read_csv(os.path.join(root, dataset, 'train.csv'))
            self.label = np.array(label['label'])
        else:
            self.filename = sorted(glob.glob(os.path.join(root, dataset, 'test', '*png')))
            label = pd.read_csv(os.path.join(root, dataset, 'test.csv'))
            self.label = np.array(label['label'])

    def __getitem__(self, index):
        if self.channel == 1:
            image = cv2.imread(self.filename[index], 0)
            image = transforms.ToTensor()(image).type(torch.float)
            #image = transforms.Normalize([0.5], [0.5])(image)
        elif self.channel == 3:
            image = cv2.imread(self.filename[index])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = transforms.ToTensor()(image).type(torch.float)
            #image = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(image)
        label = int(self.label[index])
        return image, label
    
    def __len__(self):
        return len(self.filename)

class DsnTestset(data.Dataset):
    def __init__(self, root, channel):
        self.filename = sorted(glob.glob(os.path.join(root, '*png')))
        self.channel = int(channel)
 
    def __getitem__(self, index):
        if self.channel == 1:
            image = cv2.imread(self.filename[index], 0)
            image = transforms.ToTensor()(image).type(torch.float)
            #image = transforms.Normalize([0.5], [0.5])(image)
        elif self.channel == 3:
            image = cv2.imread(self.filename[index])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = transforms.ToTensor()(image).type(torch.float)
            #image = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(image)
        
        return image
    
    def __len__(self):
        return len(self.filename)
