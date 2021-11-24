import torch
from torch.utils.data import DataLoader
import pickle
import sys
from model import Encoder50, Encoder152
from dataset import M3SDA_Dataset

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print('Device used', device)

target = sys.argv[1]
train = sys.argv[2]
file_root = sys.argv[3]
#file_root = '/home/kkeen/dlcv/final_data'
if train == 'train':
    dataset = M3SDA_Dataset(root=file_root, domain=target, train=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
else:
    dataset = M3SDA_Dataset(root=file_root, domain=target, train=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

encoder50 = Encoder50().to(device)
encoder50.eval()
encoder152 = Encoder152().to(device)
encoder152.eval()

feature = []
label = []
with torch.no_grad():
    for i, (data, l) in enumerate(dataloader):
        data = data.to(device)
        f50 = encoder50(data).squeeze()
        f152 = encoder152(data).squeeze()
        f50 = f50.cpu()
        f152 = f152.cpu()
        f = torch.cat((f50, f152), 0)
        feature.append(f)
        label.append(l.item())
print(feature[0].size())
feature_file = './feature/'+target+'_'+train+'_feature.pickle'
label_file = './feature/'+target+'_'+train+'_label.pickle'
with open(feature_file, 'wb') as f:
    pickle.dump(feature, f)
with open(label_file, 'wb') as f:
    pickle.dump(label, f)
print('finished')

