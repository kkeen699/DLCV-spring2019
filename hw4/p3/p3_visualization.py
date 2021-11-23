import matplotlib
import matplotlib.pyplot as plt
from os import listdir
import os
import torch
import torchvision
import torchvision.transforms as transforms
import cv2

gt_path = '../hw4_data/FullLengthVideos/labels/valid'
#gt_file = sorted(listdir(gt_path))
pre_path = '../pre'
#pre_file = sorted(listdir(pre_path)) 
pic_path = '../hw4_data/FullLengthVideos/videos/valid/OP03-R04-ContinentalBreakfast'
pic = sorted(listdir(pic_path)) 
gt_file = ['OP03-R04-ContinentalBreakfast.txt']
pre_file = ['OP03-R04-ContinentalBreakfast.txt']

for gt, pre in zip(gt_file, pre_file):
    with open(os.path.join(gt_path, gt),'r') as f:
        answer = [int(w.strip()) for w in f.readlines()]
    with open(os.path.join(pre_path, pre), 'r') as f:
        test = [int(w.strip()) for w in f.readlines()]
    answer = answer[350:]
    test = test[350:]
    print(len(answer))
    print(len(test))

    plt.figure(figsize=(16,4))
    ax = plt.subplot(211)
    colors = plt.cm.get_cmap('tab20',11).colors
    cmap = matplotlib.colors.ListedColormap([colors[idx] for idx in test])
    bounds = [i for i in range(len(test))]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    cb1 = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, boundaries=bounds, spacing='proportional', orientation='horizontal')
    ax.set_ylabel('Prediction')
    ax.axes.get_xaxis().set_visible(False)

    ax2 = plt.subplot(212)
    cmap = matplotlib.colors.ListedColormap([colors[idx] for idx in answer])
    bounds = [i for i in range(len(test))]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    cb2 = matplotlib.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, boundaries=bounds, spacing='proportional', orientation='horizontal')
    ax2.set_ylabel('GroundTruth')
    ax2.axes.get_xaxis().set_visible(False)

    #plt.savefig('./p3_v.png')
    #plt.show()

plt.figure(figsize=(16,4))

pic = pic[350:]
step = len(pic) // 7
idx = [0, 84, 195, 267, 355, 425, -1]

image = torch.Tensor()
for i in range(7):
    p = pic_path + '/' + pic[idx[i]]
    img = cv2.imread(p)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    image = torch.cat((image, img), 0)
torchvision.utils.save_image(image.data, './p3_pic.png',nrow=7)
#print(answer[0], answer[84], answer[195],answer[267],answer[355],answer[425],answer[-1])