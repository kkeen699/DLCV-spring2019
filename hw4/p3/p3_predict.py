import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import sys
import glob
import os
from os import listdir
import cv2
import numpy as np
from p3_model import LSTM
from p3_model import ResNet_feature_extractor


def normalize(image):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_input = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad((0,40), fill=0, padding_mode='constant'),
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize
        ])
    return transform_input(image)

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print('Device used', device)

video_path = sys.argv[1]
output_path = sys.argv[2]

category_list = sorted(listdir(video_path))

feature_extractor = ResNet_feature_extractor().to(device)
feature_extractor.eval()
lstm = LSTM().to(device)
lstm.load_state_dict(torch.load('./model/p3_model.pth'))
lstm.eval()

with torch.no_grad():
    for category in category_list:
        image_list_per_folder = sorted(listdir(os.path.join(video_path,category)))
        category_frames = torch.Tensor()
        for image in image_list_per_folder:
            image = cv2.imread(os.path.join(video_path, category,image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = normalize(image)
            image = image.unsqueeze(0).to(device)
            feature = feature_extractor(image).cpu()
            category_frames = torch.cat((category_frames, feature), 0)
        length = len(category_frames)
        category_frames = category_frames.unsqueeze(0).to(device)
        output = lstm(category_frames, [length]).squeeze()
        pre = torch.max(output.data, 1)
        pre = pre[1].to('cpu')

        output_file = os.path.join(output_path, category+'.txt')
        with open(output_file, 'w') as f:
            for n in pre:
                f.write(str(n.item()) + '\n')
        print(category, 'finished')


        
