import torch
import torch.nn as nn
from torch.autograd import Function

class DANN(nn.Module):
    def __init__(self):
        super(DANN, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.Conv2d(32, 48, kernel_size=5),
            nn.BatchNorm2d(48),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(True)
        )

        self.class_classifier = nn.Sequential(
            nn.Linear(48*4*4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Dropout2d(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 10)
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(48*4*4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 2)
        )


    def forward(self, x, alpha):
        feature = self.feature(x)
        feature = feature.view(-1, 48*4*4)
        reverse_feature = ReverseLayerF.apply(feature, alpha)

        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

if __name__ == '__main__':
    model = DANN()
    model.eval()
    x = torch.rand(1, 3, 28, 28)
    alpha = 1
    c, d = model(x=x, alpha=alpha)
    print(c.size(), d.size())
