import torch.nn as nn
import torchvision

class Encoder50(nn.Module):
    def __init__(self):
        super(Encoder50, self).__init__()
        model = torchvision.models.resnet50(pretrained=True)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class Encoder152(nn.Module):
    def __init__(self):
        super(Encoder152, self).__init__()
        model = torchvision.models.resnet50(pretrained=True)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class MomentMatching(nn.Module):
    def __init__(self):
        super(MomentMatching, self).__init__()
        self.linear1 = nn.Linear(2048*2, 2048)
        self.linear2 = nn.Linear(2048, 2048)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.bn = nn.BatchNorm1d(2048)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(2048, 1024)
        self.linear2 = nn.Linear(1024, 345)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.bn = nn.BatchNorm1d(1024)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class Weighted(nn.Module):
    def __init__(self):
        super(Weighted, self).__init__()
        self.linear1 = nn.Linear(2048, 1024)
        self.linear2 = nn.Linear(1024, 3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.bn = nn.BatchNorm1d(1024)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
