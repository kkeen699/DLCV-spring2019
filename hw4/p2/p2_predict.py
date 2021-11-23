import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import sys
import pandas as pd
import os
import numpy as np
from reader import readShortVideo
from p2_model import LSTM
from p2_model import ResNet_feature_extractor


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
lstm = LSTM().to(device)
lstm.load_state_dict(torch.load('./model/p2_model.pth'))
lstm.eval()

pre_all = []
with torch.no_grad():
    for name, category in zip(video_name, video_category):
        frames = readShortVideo(video_path, category, name, downsample_factor=6, rescale_factor=1)
        frames = frames.to(device)
        feature = feature_extractor(frames)

        feature = nn.utils.rnn.pad_sequence([feature])
        length = [len(feature)]

        output, _ = lstm(feature, length)
        pre = torch.max(output.data, 1)
        pre = pre[1].to('cpu')
        pre_all.append(pre)

output_file = os.path.join(output_path, 'p2_result.txt')
with open(output_file, 'w') as f:
    for n in pre_all:
        f.write(str(n.item()) + '\n')
