import torch
from torch.utils.data import DataLoader
import os
import sys
import glob
import numpy as np
import pandas as pd
from adda_model import Encoder, Classifier
from adda_dataset import AddaTestset

file_root = sys.argv[1]
target_domain = sys.argv[2]
output_path = sys.argv[3]

testset = AddaTestset(root=file_root)
testloader = DataLoader(testset, batch_size=200, shuffle=False, num_workers=1)

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print('Device used', device)

encoder = Encoder()
classifier = Classifier()
if target_domain == 'mnistm':
    encoder.load_state_dict(torch.load('./models/adda_u_m.pth', map_location=device))
    classifier.load_state_dict(torch.load('./models/classifier_usps.pth', map_location=device))
elif target_domain == 'usps':
    encoder.load_state_dict(torch.load('./models/adda_s_u.pth', map_location=device))
    classifier.load_state_dict(torch.load('./models/classifier_svhn.pth', map_location=device))
elif target_domain == 'svhn':
    encoder.load_state_dict(torch.load('./models/adda_m_s.pth', map_location=device))
    classifier.load_state_dict(torch.load('./models/classifier_mnistm.pth', map_location=device))


encoder = encoder.to(device)
classifier = classifier.to(device)

encoder.eval()
classifier.eval()
label = np.array([])
for _, t_img in enumerate(testloader):
    t_img = t_img.to(device)
    class_output = classifier(encoder(t_img))
    pre = torch.max(class_output.data, 1)
    pre = pre[1].to('cpu')
    pre = np.array(pre)
    pre = pre.astype(int)
    label = np.append(label, pre)


filename = sorted(glob.glob(os.path.join(file_root,'*.png')))
for i in range(len(filename)):
    name = filename[i].split('/')
    filename[i] = name[-1]
filename = np.array(filename).reshape(-1, 1)
label = np.array(label).reshape(-1,1)
output = np.append(filename, label, axis=1)

d = pd.DataFrame(output, columns=['image_name', 'label'])
d['label'] = pd.to_numeric(d['label']).astype(int)

d.to_csv(output_path, index=0)
