import random
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
from dsn_model import DSN
from dsn_dataset import DsnDataset
from functions import SIMSE, DiffLoss, MSE
from torch.utils.data import DataLoader
import sys

def test(model, dataloader):
    model.eval()
    n_correct = 0
    with torch.no_grad():
        for _, (t_img, t_label) in enumerate(dataloader):
            t_img, t_label = t_img.to(device), t_label.to(device)
            result = model(input_data=t_img, mode='source', rec_scheme='share')
            pred = torch.max(result[3].data, 1)
            n_correct += (pred[1] == t_label).sum().item()

    accu = float(n_correct) / len(dataloader.dataset) * 100
    return accu


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print('Device used', device)

lr = 1e-2
image_size = 28
n_epoch = 100
step_decay_weight = 0.95
lr_decay_step = 20000
active_domain_loss_step = 10000
weight_decay = 1e-6
alpha_weight = 0.01
beta_weight = 0.075
gamma_weight = 0.25
momentum = 0.9


source = sys.argv[1]
channel = sys.argv[2]
target = ''
if source == 'mnistm':
    target = 'svhn'
elif source == 'usps':
    target = 'mnistm'
elif source == 'svhn':
    target = 'usps'

BATCH_SIZE = 32
file_root = '../hw3_data/digits'
source_trainset = DsnDataset(root=file_root, dataset=source, train=True, channel=channel)
source_trainloader = DataLoader(source_trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
target_trainset = DsnDataset(root=file_root, dataset=target, train=True, channel=channel)
target_trainloader = DataLoader(target_trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

source_testset = DsnDataset(root=file_root, dataset=source, train=False, channel=channel )
source_testloader = DataLoader(source_testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
target_testset = DsnDataset(root=file_root, dataset=target, train=False, channel=channel)
target_testloader = DataLoader(target_testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

model = DSN(channel=channel).to(device)

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

loss_classification = torch.nn.CrossEntropyLoss()
loss_recon1 = MSE()
loss_recon2 = SIMSE()
loss_diff = DiffLoss()
loss_similarity = torch.nn.CrossEntropyLoss()



#for p in model.parameters():
#    p.requires_grad = True


len_dataloader = min(len(source_trainloader), len(target_trainloader))
dann_epoch = np.floor(active_domain_loss_step / len_dataloader * 1.0)

current_step = 0
for epoch in range(n_epoch):
    print('\nepoch',  epoch+1, '=======================================')
    if epoch == 20:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10
    if epoch == 50:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10
    if epoch == 70:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10

    data_source_iter = iter(source_trainloader)
    data_target_iter = iter(target_trainloader)
    model.train()
    i = 0

    while i < len_dataloader:

        # target data training            
        optimizer.zero_grad()

        data_target = data_target_iter.next()
        t_img, t_label = data_target[0].to(device), data_target[1].to(device)
        domain_label = torch.ones(t_img.size()[0]).long().to(device)

        loss = 0


        if current_step > active_domain_loss_step:
            p = float(i + (epoch - dann_epoch) * len_dataloader / (n_epoch - dann_epoch) / len_dataloader)
            p = 2. / (1. + np.exp(-10 * p)) - 1

            # activate domain loss
            result = model(input_data=t_img, mode='target', rec_scheme='all', p=p)
            target_privte_code, target_share_code, target_domain_label, target_rec_code = result
            target_dann = gamma_weight * loss_similarity(target_domain_label, domain_label)
            loss += target_dann
        else:
            target_dann = torch.zeros(1).float().to(device)
            result = model(input_data=t_img, mode='target', rec_scheme='all')
            target_privte_code, target_share_code, _, target_rec_code = result

        target_diff= beta_weight * loss_diff(target_privte_code, target_share_code)
        loss += target_diff
        target_mse = alpha_weight * loss_recon1(target_rec_code, t_img)
        loss += target_mse
        target_simse = alpha_weight * loss_recon2(target_rec_code, t_img)
        loss += target_simse

        loss.backward()
        optimizer.step()

       
        # source data training           
        optimizer.zero_grad()

        data_source = data_source_iter.next()
        s_img, s_label = data_source[0].to(device), data_source[1].to(device)
        domain_label = torch.zeros(s_img.size()[0]).long().to(device)

        loss = 0

        if current_step > active_domain_loss_step:
            # activate domain loss
            result = model(input_data=s_img, mode='source', rec_scheme='all', p=p)
            source_privte_code, source_share_code, source_domain_label, source_class_label, source_rec_code = result
            source_dann = gamma_weight * loss_similarity(source_domain_label, domain_label)
            loss += source_dann
        else:
            source_dann = torch.zeros(1).float().to(device)
            result = model(input_data=s_img, mode='source', rec_scheme='all')
            source_privte_code, source_share_code, _, source_class_label, source_rec_code = result

        source_classification = loss_classification(source_class_label, s_label)
        loss += source_classification

        source_diff = beta_weight * loss_diff(source_privte_code, source_share_code)
        loss += source_diff
        source_mse = alpha_weight * loss_recon1(source_rec_code, s_img)
        loss += source_mse
        source_simse = alpha_weight * loss_recon2(source_rec_code, s_img)
        loss += source_simse

        loss.backward()
        optimizer.step()

        i += 1
        current_step += 1
    print('source_classification: %f, source_dann: %f, source_diff: %f, ' \
          'source_mse: %f, source_simse: %f, target_dann: %f, target_diff: %f, ' \
          'target_mse: %f, target_simse: %f' \
          % (source_classification.data.cpu().numpy(), source_dann.data.cpu().numpy(), source_diff.data.cpu().numpy(),
             source_mse.data.cpu().numpy(), source_simse.data.cpu().numpy(), target_dann.data.cpu().numpy(),
             target_diff.data.cpu().numpy(),target_mse.data.cpu().numpy(), target_simse.data.cpu().numpy()))

    # print 'step: %d, loss: %f' % (current_step, loss.cpu().data.numpy())
    print('Source accuracy:', test(model, source_testloader))
    print('Target accuracy:', test(model, target_testloader))

    if (epoch+1) % 5 == 0:
        if source == 'mnistm':
            torch.save(model.state_dict(), './m_s/epoch_' + str(epoch+1) + '.pth')
        elif source == 'usps':
            torch.save(model.state_dict(), './u_m/epoch_' + str(epoch+1) + '.pth')
        elif source == 'svhn':
            torch.save(model.state_dict(), './s_u/epoch_' + str(epoch+1) + '.pth')
