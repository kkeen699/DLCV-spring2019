import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
from torch.utils.data import DataLoader
from adda_model import Encoder, Classifier, Discriminator
from adda_dataset import AddaDataset

def test(encoder, classifier, dataloader):
    encoder.eval()
    classifier.eval()
    n_correct = 0
    with torch.no_grad():
        for _, (t_img, t_label) in enumerate(dataloader):
            t_img, t_label = t_img.to(device), t_label.to(device)
            class_output = classifier(encoder(t_img))
            pred = torch.max(class_output.data, 1)
            n_correct += (pred[1] == t_label).sum().item()

    accu = float(n_correct) / len(dataloader.dataset) * 100
    return accu


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print('Device used', device)

BATCH_SIZE = 128

source = sys.argv[1]
target = ''
src_encoder = Encoder().to(device)
src_encoder.eval()
classifier = Classifier().to(device)
classifier.eval()
tgt_encoder = Encoder().to(device)

if source == 'mnistm':
    src_encoder.load_state_dict(torch.load('./m/encoder_mnistm.pth'))
    classifier.load_state_dict(torch.load('./m/classifier_mnistm.pth'))
    tgt_encoder.load_state_dict(torch.load('./m/encoder_mnistm.pth'))
    target = 'svhn'
elif source == 'usps':
    src_encoder.load_state_dict(torch.load('./u/encoder_usps.pth'))
    classifier.load_state_dict(torch.load('./u/classifier_usps.pth'))
    tgt_encoder.load_state_dict(torch.load('./u/encoder_usps.pth'))
    target = 'mnistm'
elif source == 'svhn':
    src_encoder.load_state_dict(torch.load('./s/encoder_svhn.pth'))
    classifier.load_state_dict(torch.load('./s/classifier_svhn.pth'))
    tgt_encoder.load_state_dict(torch.load('./s/encoder_svhn.pth'))
    target = 'usps'

file_root = '../hw3_data/digits'
source_trainset = AddaDataset(root=file_root, dataset=source, train=True)
source_trainloader = DataLoader(source_trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
target_trainset = AddaDataset(root=file_root, dataset=target, train=True)
target_trainloader = DataLoader(target_trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

target_testset = AddaDataset(root=file_root, dataset=target, train=False)
target_testloader = DataLoader(target_testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)


critic = Discriminator().to(device)

criterion = nn.CrossEntropyLoss()

optimizer_tgt = optim.Adam(tgt_encoder.parameters(), lr=0.0001, betas=(0.5, 0.9))
optimizer_critic = optim.Adam(critic.parameters(), lr=0.0001, betas=(0.5, 0.9))

len_dataloader = min(len(source_trainloader), len(target_trainloader))

for epoch in range(2000):
    print('\nepoch',  epoch+1, '=======================================')
    if epoch == 20:
        for param_group in optimizer_tgt.param_groups:
            param_group['lr'] /= 10
        for param_group in optimizer_critic.param_groups:
            param_group['lr'] /= 10
    if epoch == 40:
        for param_group in optimizer_tgt.param_groups:
            param_group['lr'] /= 10
        for param_group in optimizer_critic.param_groups:
            param_group['lr'] /= 10

    data_zip = enumerate(zip(source_trainloader, target_trainloader))
    tgt_encoder.train()
    critic.train()
    for idx, ((images_src, _), (images_tgt, _)) in data_zip:
        # train discriminator
        images_src = images_src.to(device)
        images_tgt = images_tgt.to(device)

        optimizer_critic.zero_grad()

        feat_src = src_encoder(images_src)
        feat_tgt = tgt_encoder(images_tgt)
        feat_concat = torch.cat((feat_src, feat_tgt), 0)

        pred_concat = critic(feat_concat.detach())

        label_src = torch.ones(feat_src.size(0)).long()
        label_tgt = torch.zeros(feat_tgt.size(0)).long()
        label_concat = torch.cat((label_src, label_tgt), 0).to(device)

        loss_critic = criterion(pred_concat, label_concat)
        loss_critic.backward()
        optimizer_critic.step()

        # train target encoder
        for _ in range(2):
            optimizer_critic.zero_grad()
            optimizer_tgt.zero_grad()

            feat_tgt = tgt_encoder(images_tgt)

            pred_tgt = critic(feat_tgt)

            label_tgt = torch.ones(feat_tgt.size(0)).long().to(device)

            loss_tgt = criterion(pred_tgt, label_tgt)
            loss_tgt.backward()
            optimizer_tgt.step()

        if idx % (len_dataloader//5) == 0:
            print('Epoch [%d/%d], Iter [%d/%d], Loss_D: %.4f, Loss_T: %.4f'
                % (epoch+1, 100, idx, len_dataloader, loss_critic.item(), loss_tgt.item()))
    
    print('Targt accuracy:', test(tgt_encoder, classifier, target_testloader))
    if (epoch+1) % 5 == 0:
        if source == 'mnistm':
            torch.save(tgt_encoder.state_dict(), './m_s/epoch_' + str(epoch+1) + '.pth')
        elif source == 'usps':
            torch.save(tgt_encoder.state_dict(), './u_m/epoch_' + str(epoch+1) + '.pth')
        elif source == 'svhn':
            torch.save(tgt_encoder.state_dict(), './s_u/epoch_' + str(epoch+1) + '.pth')
