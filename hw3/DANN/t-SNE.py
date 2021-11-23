import torch
from torch.utils.data import DataLoader
import os
import sys
import glob
import numpy as np
import pandas as pd
from dann_model import DANN
from dann_dataset import DannDataset
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

source = sys.argv[1]
target = ''
if source == 'mnistm':
    target = 'svhn'
elif source == 'usps':
    target = 'mnistm'
elif source == 'svhn':
    target = 'usps'

file_root = '../hw3_data/digits'

model = DANN()
if target == 'mnistm':
    model.load_state_dict(torch.load('../models/usps_mnistm.pth', map_location='cpu'))
elif target == 'usps':
    model.load_state_dict(torch.load('../models/svhn_usps.pth', map_location='cpu'))
elif target == 'svhn':
    model.load_state_dict(torch.load('../models/mnistm_svhn.pth', map_location='cpu'))

#source_dataset = DannDataset(root=file_root, dataset=source, train=False)
#source_loader = DataLoader(source_dataset, batch_size=200, shuffle=False, num_workers=1)
target_dataset = DannDataset(root=file_root, dataset=target, train=False)
target_loader = DataLoader(target_dataset, batch_size=200, shuffle=False, num_workers=1)

model.eval()
feature_extracter = model.feature
fc = model.class_classifier
n_correct = 0
with torch.no_grad():
    for _, (t_img, t_label) in enumerate(target_loader):
        t_img, t_label = t_img, t_label
        f = feature_extracter(t_img).view(-1, 48*4*4)
        class_output = fc(f)
        pred = torch.max(class_output.data, 1)
        n_correct += (pred[1] == t_label).sum().item()

accu = float(n_correct) / len(target_loader.dataset) * 100
print(accu)



output = np.empty(shape=[0, 48*4*4])
class_label = np.empty(shape=[0, 1])
'''
for _, (t_img, t_label) in enumerate(source_loader):
    f = feature_extracter(t_img).view(-1, 48*4*4)
    f = np.array(f.detach())
    output = np.append(output, f, axis=0)
    t_label = np.array(t_label)
    t_label = t_label.astype(int)
    class_label = np.append(class_label, t_label)

n_correct = 0
for _, (t_img, t_label) in enumerate(target_loader):
    f = feature_extracter(t_img).view(-1, 48*4*4)
    f = np.array(f.detach())
    output = np.append(output, f, axis=0)
    t_label = np.array(t_label)
    t_label = t_label.astype(int)
    class_label = np.append(class_label, t_label)


domain_label = np.zeros((len(source_loader.dataset)))
domain_label = np.append(domain_label, np.ones((len(target_loader.dataset))))
domain_label = domain_label.astype(int)

tsne = TSNE(n_components=2, random_state=0)
X_2d = tsne.fit_transform(output)


target_names = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
target_ids = range(len(target_names))

plt.figure(figsize=(6, 5))
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'grey', 'orange', 'purple']
for i, c, label in zip(target_ids, colors, target_names):
    plt.scatter(X_2d[class_label == i, 0], X_2d[class_label == i, 1], c=c, label=label)
plt.legend()
print('done')
plt.show()

target_names = np.array(['source domain', 'target domain'])
target_ids = range(len(target_names))

plt.figure(figsize=(6, 5))
colors = ['r', 'b']
for i, c, label in zip(target_ids, colors, target_names):
    plt.scatter(X_2d[domain_label == i, 0], X_2d[domain_label == i, 1], c=c, label=label)
plt.legend()
print('done')
plt.show()
'''
'''
from sklearn import datasets
digits = datasets.load_digits()
# Take the first 500 data points: it's hard to see 1500 points
X = digits.data[:10]
y = digits.target[:10]

############################################################
# Fit and transform with a TSNE
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)

############################################################
# Project the data in 2D
X_2d = tsne.fit_transform(X)

############################################################
# Visualize the data
target_ids = range(len(digits.target_names))
print(type(digits.target_names))

from matplotlib import pyplot as plt
plt.figure(figsize=(6, 5))
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
for i, c, label in zip(target_ids, colors, digits.target_names):
    plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)
plt.legend()
plt.show()

'''