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

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print('Device used', device)

video_path = sys.argv[1]
label_file = sys.argv[2]
output_path = sys.argv[3]


df = pd.read_csv(label_file)
#action_labels = df['Action_labels'].tolist()
video_name = df['Video_name'].tolist()
video_category = df['Video_category'].tolist()



feature_extractor = ResNet_feature_extractor().to(device)
feature_extractor.eval()
clf = Classifier().to(device)
clf.load_state_dict(torch.load('./model/p1_model.pth'))
clf.eval()

pre_all = []
with torch.no_grad():
    for name, category in zip(video_name, video_category):
        frames = readShortVideo(video_path, category, name, downsample_factor=6, rescale_factor=1)
        frames = frames.to(device)
        feature = feature_extractor(frames)
        feature = torch.mean(feature, 0).unsqueeze(0)

        output = clf(feature)
        pre = torch.max(output.data, 1)
        pre = pre[1].to('cpu')
        pre_all.append(pre)
        
output_file = os.path.join(output_path, 'p1_valid.txt')
with open(output_file, 'w') as f:
    for n in pre_all:
        f.write(str(n.item()) + '\n')
