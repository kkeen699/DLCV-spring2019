import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import sys
import pandas as pd
import os
import numpy as np
from reader import readShortVideo
from p1_model import Classifier
from p1_model import ResNet_feature_extractor
import pickle
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print('Device used', device)

video_path = sys.argv[1]
label_file = sys.argv[2]
output_path = sys.argv[3]


with open('./features/valid_feature.pickle', 'rb') as f:
    features = pickle.load(f)
with open('./features/valid_label.pickle', 'rb') as f:
    labels = pickle.load(f)



feature_extractor = ResNet_feature_extractor().to(device)
feature_extractor.eval()
clf = Classifier().to(device)
clf.load_state_dict(torch.load('./model/epoch_100.pth'))
clf.eval()

pre_all = []
n_correct = 0
with torch.no_grad():
    for feature, label in zip(features, labels):
        label = torch.tensor(label)
        feature, label = feature.unsqueeze(0).to(device), label.to(device)
        output = clf(feature)
        pre = torch.max(output.data, 1)
        n_correct += (pre[1] == label).sum().item()
        pre = pre[1].to('cpu')
        pre_all.append(pre)
acc = float(n_correct)/len(labels) *100
print(acc)

output_file = os.path.join(output_path, 'p1_valid.txt')
with open(output_file, 'w') as f:
    for n in pre_all:
        f.write(str(n.item()) + '\n')

