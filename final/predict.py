import torch
import sys
import cv2
import os
import glob
import pandas as pd
import numpy as np
from dataset import normalize
from model import Encoder50, Encoder152, Classifier, MomentMatching

#root = '/home/kkeen/dlcv/final_data/test'
target = sys.argv[1]
root = sys.argv[2]
output_file = sys.argv[3]

filename = sorted(glob.glob(os.path.join(root,'*.jpg')))

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print('Device used:', device)

encoder50 = Encoder50().to(device)
encoder50.eval()
encoder152 = Encoder152().to(device)
encoder152.eval()
matching = MomentMatching().to(device)
matching.load_state_dict(torch.load('./model_final/'+target+'/matching_200.pth'))
matching.eval()
clf1 = Classifier().to(device)
clf1.load_state_dict(torch.load('./model_final/'+target+'/clf0_200.pth'))
clf1.eval()
clf2 = Classifier().to(device)
clf2.load_state_dict(torch.load('./model_final/'+target+'/clf1_200.pth'))
clf2.eval()
clf3 = Classifier().to(device)
clf3.load_state_dict(torch.load('./model_final/'+target+'/clf2_200.pth'))
clf3.eval()



label = np.array([])
for n in filename:
    image = cv2.imread(n)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224,224), interpolation=cv2.INTER_CUBIC)
    image = normalize(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    with torch.no_grad():
        f1 = encoder50(image).squeeze()
        f2 = encoder152(image).squeeze()
        feature = torch.cat((f1,f2), 0).unsqueeze(0)
        output = matching(feature)
        pre = (clf1(output) + clf2(output) + clf3(output)) / 3
        pre = torch.max(pre.data, 1)
        pre = pre[1].to('cpu')
        pre = np.array(pre)
        pre = pre.astype(int)
        label = np.append(label, pre)


for i in range(len(filename)):
    name = filename[i].split('/')
    filename[i] = name[-2] + '/' + name[-1]
filename = np.array(filename).reshape(-1, 1)
label = np.array(label).reshape(-1,1)
output = np.append(filename, label, axis=1)

d = pd.DataFrame(output, columns=['image_name', 'label'])
d['label'] = pd.to_numeric(d['label']).astype(int)

#d.to_csv('./real_pre_2.csv', index=0)
d.to_csv(output_file, index=0)
