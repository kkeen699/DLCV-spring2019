import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
from torch.utils.data import DataLoader
from adda_model import Encoder, Classifier
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
file_root = '../hw3_data/digits'
source_trainset = AddaDataset(root=file_root, dataset=source, train=True)
source_trainloader = DataLoader(source_trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

source_testset = AddaDataset(root=file_root, dataset=source, train=False)
source_testloader = DataLoader(source_testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

encoder = Encoder().to(device)
classifier = Classifier().to(device)

optimizer = optim.Adam(
    list(encoder.parameters()) +list(classifier.parameters()),
    lr=0.0001, betas=(0.5, 0.9))
criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    print('\nepoch',  epoch+1, '=======================================')
    encoder.train()
    classifier.train()
    for idx, (images, labels) in enumerate(source_trainloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        preds = classifier(encoder(images))
        loss = criterion(preds, labels)

        loss.backward()
        optimizer.step()

        if idx % (len(source_trainloader)//5) == 0:
            print('Epoch [%d/%d], Iter [%d/%d], Loss: %.4f'
                % (epoch+1, 100, idx, len(source_trainloader), loss.item()))
    
    print('Source accuracy:', test(encoder, classifier, source_testloader))
    if (epoch+1) % 5 == 0:
        if source == 'usps':
            torch.save(encoder.state_dict(), './u/encoder_epoch_%d.pth' % (epoch+1))
            torch.save(classifier.state_dict(), './u/classifier_epoch_%d.pth' % (epoch+1))
        elif source == 'mnistm':
            torch.save(encoder.state_dict(), './m/encoder_epoch_%d.pth' % (epoch+1))
            torch.save(classifier.state_dict(), './m/classifier_epoch_%d.pth' % (epoch+1))
        elif source == 'svhn':
            torch.save(encoder.state_dict(), './s/encoder_epoch_%d.pth' % (epoch+1))
            torch.save(classifier.state_dict(), './s/classifier_epoch_%d.pth' % (epoch+1))