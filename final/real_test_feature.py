import torch
from torch.utils.data import DataLoader
import pickle
from model import Encoder
from dataset import RealDataset

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print('Device used', device)


dataset = RealDataset()
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)


encoder = Encoder().to(device)
encoder.eval()

feature = []
label = []
with torch.no_grad():
    for i, data in enumerate(dataloader):
        data = data.to(device)
        f = encoder(data).squeeze()
        f = f.cpu()
        feature.append(f)
        

feature_file = './feature/real_test_feature_50.pickle'
#label_file = './feature/'+target+'_'+train+'_label.pickle'
with open(feature_file, 'wb') as f:
    pickle.dump(feature, f)
#with open(label_file, 'wb') as f:
#    pickle.dump(label, f)
print('finished')

