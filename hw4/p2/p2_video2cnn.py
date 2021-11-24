import torch
from torch.utils.data import DataLoader
import pickle
from p2_model import ResNet_feature_extractor
from p2_dataset import VideoDataset

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print('Device used', device)

trainset = VideoDataset(t_v='train')
trainloader = DataLoader(trainset, batch_size=1, shuffle=False, num_workers=1)
validset = VideoDataset(t_v='valid')
validloader = DataLoader(validset, batch_size=1, shuffle=False, num_workers=1)

feature_extractor = ResNet_feature_extractor().to(device)
feature_extractor.eval()

train_feature = []
train_label = []
with torch.no_grad():
    for i, (data, label) in enumerate(trainloader):
        data = data.to(device).squeeze()
        feature = feature_extractor(data).cpu()
        train_feature.append(feature)
        train_label.append(label.item())
print(train_feature[0].size())
with open('./features/train_feature.pickle', 'wb') as f:
    pickle.dump(train_feature, f)
with open('./features/train_label.pickle', 'wb') as f:
    pickle.dump(train_label, f) 
print('train feature finished')

valid_feature = []
valid_label = []
with torch.no_grad():
    for i, (data, label) in enumerate(validloader):
        data = data.to(device).squeeze()
        feature = feature_extractor(data).cpu()
        valid_feature.append(feature)
        valid_label.append(label.item())
print(valid_feature[0].size())
with open('./features/valid_feature.pickle', 'wb') as f:
    pickle.dump(valid_feature, f)
with open('./features/valid_label.pickle', 'wb') as f:
    pickle.dump(valid_label, f)
print('validation feature finished')

