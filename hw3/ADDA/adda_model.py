import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 48, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU()
        )

        self.fc1 = nn.Linear(48*4*4, 500)
    
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 48*4*4)
        x = self.fc1(x)
        return x

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.fc = nn.Linear(500, 10)
    
    def forward(self, x):
        x = F.dropout(F.relu(x), training=self.training)
        x = self.fc(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2),
            nn.LogSoftmax()
        )

    def forward(self, x):
        x = self.layer(x)
        return x
