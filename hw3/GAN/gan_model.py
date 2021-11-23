import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, f=64):
        super(Generator, self).__init__()
        self.decoder = nn.Sequential(
            # input is n x 100 x 1 x 1
            nn.ConvTranspose2d( 100, f*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(f * 8),
            nn.ReLU(inplace=True),
            # state size is n x (f*8) x 4 x 4
            nn.ConvTranspose2d(f*8, f*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f * 4),
            nn.ReLU(inplace=True),
            # state size is n x (f*4) x 8 x 8
            nn.ConvTranspose2d(f*4, f*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f * 2),
            nn.ReLU(inplace=True),
            # state size is n x (f*2) x 16 x 16
            nn.ConvTranspose2d(f*2, f, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f),
            nn.ReLU(inplace=True),
            # state size is n x (f) x 32 x 32
            nn.ConvTranspose2d(f, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size is n x 3 x 64 x 64
        )

    def forward(self, X):
        output = self.decoder(X)/2.0+0.5
        return output

class Discriminator(nn.Module):
    def __init__(self, f=64):
        super(Discriminator, self).__init__()
        self.decoder = nn.Sequential(
            # input is n x 3 x 64 x 64
            nn.Conv2d(3, f, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size is n x f x 32 x 32
            nn.Conv2d(f, f*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size is n x (f*2) x 16 x 16
            nn.Conv2d(f*2, f*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size is n x (f*4) x 8 x 8
            nn.Conv2d(f*4, f*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size is n x (f*8) x 4 x 4
            nn.Conv2d(f*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.decoder(x)
        return output.view(-1, 1)

if __name__ == '__main__':
    g = Generator()
    x = torch.rand(1,100,1,1)
    y = g(x)
    print(x.size())
    print(y.size())
    x = torch.rand(2,3,64,64)
    d = Discriminator()
    y = d(x)
    print(x.size())
    print(y.size())
