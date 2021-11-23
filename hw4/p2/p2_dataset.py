import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data import DataLoader
import os
from os import listdir
import pickle
import pandas as pd
import numpy as np
from reader import readShortVideo

class VideoDataset(data.Dataset):
    def __init__(self, t_v): 
        df = pd.read_csv('../hw4_data/TrimmedVideos/label/gt_' + t_v + '.csv')
        self.action_labels = df['Action_labels'].tolist()
        self.video_name = df['Video_name'].tolist()
        self.video_category = df['Video_category'].tolist()
        self.video_path = '../hw4_data/TrimmedVideos/video/' + t_v + '/'
    
    def __getitem__(self, idx):
        frames = readShortVideo(self.video_path, self.video_category[idx], self.video_name[idx], downsample_factor=6, rescale_factor=1)
        label = self.action_labels[idx]

        return frames, label
    
    def __len__(self):
        return len(self.action_labels)


class CNN_feature(data.Dataset):
    def __init__(self, t_v):
        if t_v == 'train':
            with open('./features/train_feature.pickle', 'rb') as f:
                self.features = pickle.load(f)
            with open('./features/train_label.pickle', 'rb') as f:
                self.labels = pickle.load(f) 
        
        if t_v == 'valid':
            with open('./features/valid_feature.pickle', 'rb') as f:
                self.features = pickle.load(f)
            with open('./features/valid_label.pickle', 'rb') as f:
                self.labels = pickle.load(f)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        return feature, label
    
    def __len__(self):
        return len(self.features)

