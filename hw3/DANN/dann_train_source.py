import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
from torch.utils.data import DataLoader
from dann_model import DANN
from dann_dataset import DannDataset

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print('Device used', device)

source = sys.argv[1]
target = sys.argv[2]

BATCH_SIZE = 128
N_EPOCH = 100

file_root = '../hw3_data/digits'
source_trainset = DannDataset(root=file_root, dataset=source, train=True)
source_trainloader = DataLoader(source_trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
target_trainset = DannDataset(root=file_root, dataset=target, train=True)
target_trainloader = DataLoader(target_trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

source_testset = DannDataset(root=file_root, dataset=source, train=False)
source_testloader = DataLoader(source_testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
target_testset = DannDataset(root=file_root, dataset=target, train=False)
target_testloader = DataLoader(target_testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

model = DANN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

class_criterion = nn.CrossEntropyLoss()

def test(model, dataloader):
    alpha = 0
    model.eval()
    n_correct = 0
    with torch.no_grad():
        for _, (t_img, t_label) in enumerate(dataloader):
            t_img, t_label = t_img.to(device), t_label.to(device)
            class_output, _ = model(x=t_img, alpha=alpha)
            pred = torch.max(class_output.data, 1)
            n_correct += (pred[1] == t_label).sum().item()

    accu = float(n_correct) / len(dataloader.dataset) * 100
    return accu

for epoch in range(1, N_EPOCH+1):
    print('\nepoch',  epoch, '=======================================')
    model.train()
    if epoch == 30:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10
    if epoch == 50:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10

    len_dataloader = len(source_trainloader)
    data_src_iter = iter(source_trainloader)

    i = 1
    while i < len_dataloader + 1:
        p = float(i + epoch * len_dataloader) / N_EPOCH / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # Training model using source data
        data_source = data_src_iter.next()
        optimizer.zero_grad()
        s_img, s_label = data_source[0].to(device), data_source[1].to(device)
        domain_label = torch.zeros(s_img.size()[0]).long().to(device)
        class_output, _ = model(x=s_img, alpha=alpha)
        err_s_label = class_criterion(class_output, s_label)

        err_s_label.backward()
        optimizer.step()

        if i % (len_dataloader//5) == 0:
            print('Epoch [%d/%d], Iter [%d/%d], Loss: %.4f'
                % (epoch, N_EPOCH, i, len_dataloader, err_s_label.item()))
        
        i += 1
    print('Source accuracy:', test(model, source_testloader))
    print('Target accuracy:', test(model, target_testloader))
    if (epoch) % 5 == 0:
        torch.save(model.state_dict(), './s/model_epoch_%d.pth' % (epoch))


