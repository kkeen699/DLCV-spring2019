import torch
import torch.nn as nn
import torchvision

class ResNet_feature_extractor(nn.Module):
    def __init__(self):
        super(ResNet_feature_extractor, self).__init__()
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

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        
        self.linear1 = nn.Linear(2048, 1024)
        self.linear2 = nn.Linear(1024, 11)
        self.relu = nn.ReLU()
        self.bn_1 = nn.BatchNorm1d(1024)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn_1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

if __name__ == '__main__':
    x = torch.rand(2,3,224,224)
    model = ResNet_feature_extractor()
    clf = Classifier()
    y = model(x)
    print(y.size())
    pre = clf(y)
    print(pre.size())