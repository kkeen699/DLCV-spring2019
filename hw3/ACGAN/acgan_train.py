import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from acgan_model import Generator, Discriminator
from acgan_dataset import ACGanDataset

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print('Device used:', device)

BATCH_SIZE = 64

torch.manual_seed(6)
up = np.ones(10)
down = np.zeros(10)
fixed_class = np.hstack((up,down))
fixed_class = torch.from_numpy(fixed_class).view(20,1,1,1).type(torch.FloatTensor)
fixed_noise = torch.randn(10, 100, 1, 1)
fixed_noise = torch.cat((fixed_noise,fixed_noise))
fixed_noise = torch.cat((fixed_noise, fixed_class),1).to(device)

file_root = '../hw3_data/face'
trainset = ACGanDataset(root=file_root, feature='Smiling')
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

G = Generator().to(device)
D = Discriminator().to(device)

optimizerG = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerD = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

score_criterion = nn.BCELoss()
feature_criterion = nn.BCELoss()

for epoch in range(200):
    print('\nepoch',  epoch+1, '=======================================')
    if (epoch+1) == 21:
        for param_group in optimizerG.param_groups:
            param_group['lr'] /= 2
        for param_group in optimizerD.param_groups:
            param_group['lr'] /= 2
    if (epoch+1) == 51:
        for param_group in optimizerG.param_groups:
            param_group['lr'] /= 10
        for param_group in optimizerD.param_groups:
            param_group['lr'] /= 10

    G.train()
    D.train()
    for i, (data, label_real, label_feature) in enumerate(trainloader):
        data = data.to(device)
        label_real = label_real.to(device)
        label_feature = label_feature.to(device)
        # train Discriminator
        optimizerD.zero_grad()

        ## real image
        output_score, output_feature = D(data)
        D_real_loss = (score_criterion(output_score, label_real) + feature_criterion(output_feature, label_feature))/2

        ## fake image
        noise = torch.randn(BATCH_SIZE, 100, 1, 1)
        fake_feature = torch.randint(2,(BATCH_SIZE, 1, 1, 1))
        fake_feature = fake_feature.type(torch.FloatTensor)
        noise = torch.cat((noise, fake_feature),1).to(device)
        fake_label = torch.zeros(label_real.size()).to(device)
        fake_feature = fake_feature.view(BATCH_SIZE, 1).to(device)
        fake_image = G(noise)
        output_score, output_feature = D(fake_image.detach())
        D_fake_loss = (score_criterion(output_score, fake_label) + feature_criterion(output_feature, fake_feature))/2

        ## update Disciminator
        D_train_loss = D_real_loss + D_fake_loss
        D_train_loss.backward()
        optimizerD.step()
        
        # train Generator
        for _ in range(2):
            optimizerG.zero_grad()
            noise = torch.randn(BATCH_SIZE, 100, 1, 1)
            feature = torch.randint(2,(BATCH_SIZE, 1, 1, 1))
            feature = feature.type(torch.FloatTensor)
            noise = torch.cat((noise, feature),1).to(device)
            label = torch.ones(label_real.size()).to(device)
            feature = feature.view(BATCH_SIZE, 1).to(device)
            fake_image = G(noise)
            output_score, output_feature = D(fake_image)
            G_loss = score_criterion(output_score, label) + feature_criterion(output_feature, feature)
            G_loss.backward()
            optimizerG.step()


        if (i+1) % (len(trainloader)/5) == 0:
            print('Epoch [%d/%d], Iter [%d/%d], Loss_D: %.4f Loss_G: %.4f'
                % (epoch+1, 200, i+1, len(trainloader), D_train_loss.item(), G_loss.item()))
    
    if (epoch+1) % 5 == 0:
        torch.save(G.state_dict(), './G/G_epoch_%d.pth' % (epoch+1))
        torch.save(D.state_dict(), './D/D_epoch_%d.pth' % (epoch+1))
        G.eval()
        img = G(fixed_noise)
        torchvision.utils.save_image(img.cpu().data, './ACGAN_output/fig2_2_'+str(epoch+1)+'.jpg',nrow=10)

