import torch
from torch.utils.data import DataLoader
import pickle
from p1_model import ResNet_feature_extractor
from p1_dataset import VideoDataset
#from sklearn.manifold import TSNE
#from matplotlib import pyplot as plt

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
        data = data.to(device).squeeze(0)
        feature = feature_extractor(data)
        feature = torch.mean(feature, 0).cpu()
        train_feature.append(feature)
        train_label.append(label.item())
with open('./features/train_feature.pickle', 'wb') as f:
    pickle.dump(train_feature, f)
with open('./features/train_label.pickle', 'wb') as f:
    pickle.dump(train_label, f) 
print('train feature finished')

valid_feature = []
valid_label = []
with torch.no_grad():
    for i, (data, label) in enumerate(validloader):
        data = data.to(device).squeeze(0)
        feature = feature_extractor(data)
        feature = torch.mean(feature, 0).cpu()
        valid_feature.append(feature)
        valid_label.append(label.item())
with open('./features/valid_feature.pickle', 'wb') as f:
    pickle.dump(valid_feature, f)
with open('./features/valid_label.pickle', 'wb') as f:
    pickle.dump(valid_label, f)
print('validation feature finished')
'''
tsne = TSNE(n_components=2, random_state=0)
X_2d = tsne.fit_transform(valid_feature)
target_names = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
target_ids = range(len(target_names))

plt.figure(figsize=(6, 5))
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'grey', 'orange', 'purple']
for i, c, label in zip(target_ids, colors, target_names):
    plt.scatter(X_2d[valid_label == i, 0], X_2d[valid_label == i, 1], c=c, label=label)
plt.legend()
plt.show()
plt.savefig('CNN_tsne.png')
'''
