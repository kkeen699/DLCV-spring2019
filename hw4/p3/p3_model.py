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

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=2048, hidden_size=512, num_layers=2, dropout=0.5, batch_first=True)
        self.bn_0 = nn.BatchNorm1d(512)
        self.fc_1 = nn.Linear(512, 11)
        self.relu = nn.ReLU()

    def forward(self, padded_sequence, input_lengths, hidden=None):
        packed = torch.nn.utils.rnn.pack_padded_sequence(padded_sequence, input_lengths, batch_first=True)
        output, (hn, cn) = self.lstm(packed, hidden)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = self.fc_1(output)
        return output

if __name__ == '__main__':
    a = torch.rand(1, 2, 2048)
    model = LSTM()
    pre = model(a, [2])
    print(pre.size())
