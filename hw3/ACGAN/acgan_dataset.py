import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import glob
import cv2
import os

class ACGanDataset(data.Dataset):
    def __init__(self, root, feature):
        self.filename = sorted(glob.glob(os.path.join(root, 'train', '*.png')))
        train_csv = pd.read_csv(os.path.join(root, 'train.csv'))
        self.label = np.array(train_csv[feature])
    
    def __getitem__(self, index):
        image = cv2.imread(self.filename[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image/255
        image = transforms.ToTensor()(image).type(torch.float)
        label_real = torch.ones(1)
        label_f = torch.tensor(self.label[index]).view(1)
        return image, label_real, label_f

    def __len__(self):
        return len(self.filename)
