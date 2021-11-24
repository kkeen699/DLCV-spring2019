import torch
import sys
import pickle
from model import MomentMatching
from dataset import DataFeature
from torch.utils.data import DataLoader


target = sys.argv[1]

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print('Device used:', device)


matching = MomentMatching().to(device)
matching.load_state_dict(torch.load('./model_final/'+ target +'/matching_200.pth'))
matching.eval()

domain = ['infograph', 'quickdraw', 'real', 'sketch']

for d in domain:
    feature = []
    if d != target:
        dataset = DataFeature(domain=d, train='train')
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
        with torch.no_grad():
            for i, (data,l) in enumerate(dataloader):
                data = data.to(device)
                f = matching(data).squeeze()
                f = f.cpu()
                feature.append(f)
    else:
        dataset = DataFeature(domain=d, train='test')
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                data = data.to(device)
                f = matching(data).squeeze()
                f = f.cpu()
                feature.append(f)

    feature_file = './feature/'+target+'/'+ d + '.pickle'
    with open(feature_file, 'wb') as f:
        pickle.dump(feature, f)
