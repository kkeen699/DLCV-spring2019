import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, f=64):
        super(Generator, self).__init__()
        self.decoder = nn.Sequential(
            # input is n x 101 x 1 x 1
            nn.ConvTranspose2d(101, f*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(f*8),
            nn.ReLU(inplace=True),
            # state size is n x (f*8) x 4 x 4
            nn.ConvTranspose2d(f*8, f*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f*4),
            nn.ReLU(inplace=True),
            # state size is n x (f*4) x 8 x 8
            nn.ConvTranspose2d(f*4, f*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f*2),
            nn.ReLU(inplace=True),
            # state size is n x (f*2) x 16 x 16
            nn.ConvTranspose2d(f*2, f, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f),
            nn.ReLU(inplace=True),
            # state size is n x (f) x 32 x 32
            nn.ConvTranspose2d(f, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. n x 3 x 64 x 64
        )
    def forward(self, x):
        output = self.decoder(x)/2.0+0.5
        return output

class Discriminator(nn.Module):
    def __init__(self, f=64):
        super(Discriminator, self).__init__()
        self.decoder = nn.Sequential(
            # input is n x 3 x 64 x 64
            nn.Conv2d(3, f, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # state size is n x (f) x 32 x 32
            nn.Conv2d(f, f*2, 4, 2, 1),
            nn.BatchNorm2d(f * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size is n x (f*2) x 16 x 16
            nn.Conv2d(f*2, f*4, 4, 2, 1),
            nn.BatchNorm2d(f * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size is n x (f*4) x 8 x 8
            nn.Conv2d(f*4, f*8, 4, 2, 1),
            nn.BatchNorm2d(f * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size is n x (f*8) x 4 x 4
            nn.Conv2d(f*8, f*1, 4, 1, 0)
            # state size is n x (f*1) x 1 x 1
        )

        self.fc_score = nn.Linear(f*1, 1)
        self.fc_feature = nn.Linear(f*1, 1) # one class
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        decode_output = self.decoder(x)

        flat = decode_output.view(-1,64)
        fc_score = self.fc_score(flat)
        fc_feature = self.fc_feature(flat)
        
        realfake = self.sigmoid(fc_score)
        feature = self.sigmoid(fc_feature)
        
        return realfake, feature


if __name__ == '__main__':
    g = Generator()
    x = torch.randn(1, 101, 1, 1)
    y = g(x)
    print(x.size())
    print(y.size())
    x = torch.rand(2,3,64,64)
    d = Discriminator()
    r, c = d(x)
    print(x.size())
    print(r.size())
    print(c.size())