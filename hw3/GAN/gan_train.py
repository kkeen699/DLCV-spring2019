import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from gan_model import Generator, Discriminator
from gan_dataset import GanDataset



use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print('Device used:', device)

BATCH_SIZE = 64

torch.manual_seed(6)
fixed_noise = torch.randn(32, 100, 1, 1).to(device)

file_root = '../hw3_data/face/train/'
trainset = GanDataset(root=file_root)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

G = Generator().to(device)
D = Discriminator().to(device)

optimizerG = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerD = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

criterion = nn.BCELoss()

for epoch in range(120):
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
    for i, (data, label) in enumerate(trainloader):
        data = data.to(device)
        label = label.to(device)
        # train Discriminator
        optimizerD.zero_grad()

        ## real image
        output = D(data)
        D_real_loss = criterion(output, label)

        ## fake image
        noise = torch.randn(BATCH_SIZE, 100, 1, 1).to(device)
        fake_image = G(noise)
        output = D(fake_image.detach())
        fake_label = torch.zeros(label.size()).to(device)
        D_fake_loss = criterion(output, fake_label)

        ## update D
        D_train_loss = D_real_loss + D_fake_loss
        D_train_loss.backward()
        optimizerD.step()

        # train Generator
        for _ in range(1):
            optimizerG.zero_grad()
            noise = torch.randn(BATCH_SIZE, 100, 1, 1).to(device)
            fake_image = G(noise)
            label = torch.ones(label.size()).to(device)
            output = D(fake_image)
            G_loss = criterion(output, label)
            G_loss.backward()
            optimizerG.step()

        if (i+1) % (len(trainloader)/5) == 0:
            print('Epoch [%d/%d], Iter [%d/%d], Loss_D: %.4f Loss_G: %.4f'
                % (epoch+1, 105, i+1, len(trainloader), D_train_loss.item(), G_loss.item()))
    
    if (epoch+1) % 5 == 0:
        torch.save(G.state_dict(), './G/G_epoch_%d.pth' % (epoch+1))
        torch.save(D.state_dict(), './D/D_epoch_%d.pth' % (epoch+1))
        G.eval()
        img = G(fixed_noise)
        torchvision.utils.save_image(img.cpu().data, './GAN_output/fig1_2_'+str(epoch+1)+'.jpg',nrow=8)



