import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import DataFeature
from model import Classifier

class MD2(nn.Module):
    def __init__(self):
        super(MD2, self).__init__()

    def forward(self, source, target):

        avg_source = []
        for s in source:
            avg_s = torch.sum(s, dim=0) / s.size(0)
            avg_source.append(avg_s)
        
        var_source = []
        for s in source:
            var_s = torch.sum(s**2, dim=0) / s.size(0)
            var_source.append(var_s)

        avg_target = torch.sum(target, dim=0) / target.size(0)
        var_target = torch.sum(target**2, dim=0) / target.size(0)

        # between source and target
        loss_avg_st = 0.0
        loss_var_st = 0.0
        for avg_s in avg_source:
            loss_avg_st += F.mse_loss(avg_s, avg_target, reduction='sum')
        for var_s in var_source:
            loss_var_st += F.mse_loss(var_s, var_target, reduction='sum')

        # between source and source
        loss_avg_ss = 0.0
        loss_var_ss = 0.0
        n = 0
        for i in range(len(avg_source)-1):
            for j in range(i+1, len(avg_source)):
                loss_avg_ss += F.mse_loss(avg_source[i], avg_source[j], reduction='sum')
                n += 1
        for i in range(len(var_source)-1):
            for j in range(i+1, len(var_source)):
                loss_var_ss += F.mse_loss(var_source[i], var_source[j], reduction='sum')
        
        loss = (loss_avg_st + loss_var_st) / len(source) + (loss_avg_ss + loss_var_ss) / n
        
        return loss

class MD2_2(nn.Module):
    def __init__(self):
        super(MD2_2, self).__init__()

    def forward(self, source, target):
        source = torch.cat((source[0], source[1], source[2]), 0)
        avg_source = torch.sum(source, dim=0) / source.size(0)
        var_source = torch.sum(source**2, dim=0) / source.size(0)

        avg_target = torch.sum(target, dim=0) / target.size(0)
        var_target = torch.sum(target**2, dim=0) / target.size(0)

        loss_avg = F.mse_loss(avg_source, avg_target, reduction='sum')
        loss_var = F.mse_loss(var_source, var_target, reduction='sum')
        
        #return loss_avg
        return loss_avg + loss_var

class CORAL_loss(nn.Module):
    def __init__(self):
        super(CORAL_loss, self).__init__()

    def forward(self, source, target):
        source = torch.cat((source[0], source[1], source[2]), 0)

        # source covariance
        xm = torch.mean(source, 0, keepdim=True) - source
        xc = xm.t() @ xm

        # target covariance
        xmt = torch.mean(target, 0, keepdim=True) - target
        xct = xmt.t() @ xmt

        # frobenius norm between source and target
        loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
        #loss = loss/(4*d*d)

        return loss
    

def getdata(target, BATCH_SIZE):
    data = ['infograph', 'quickdraw', 'real', 'sketch']
    print('Target', target)
    target_trainset = DataFeature(domain=target, train='train')
    target_trainloader = DataLoader(target_trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    if target == 'real':
        target_testloader = 0
    else:
        target_testset = DataFeature(domain=target, train='test')
        target_testloader = DataLoader(target_testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)


    source_loader = []
    for i, d in enumerate(data):
        if d != target:
            print('Source', i, d)
            trainset = DataFeature(domain=d, train='train')
            trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
            source_loader.append(trainloader)
    
    return source_loader, target_trainloader, target_testloader

def getclf(target, device):
    data = ['infograph', 'quickdraw', 'real', 'sketch']
    classifiers = []
    for d in data:
        if d != target:
            clf = Classifier().to(device)
            clf.load_state_dict(torch.load('./clf/'+ d +'/clf_200.pth'))
            classifiers.append(clf)
    return classifiers
    

def evaluate(matching, classifiers, testloader, device):
    matching.eval()
    for c in classifiers:
        c.eval()
    n_correct1 = 0
    with torch.no_grad():
        for i, (data, label) in enumerate(testloader):
            data, label = data.to(device), label.to(device)
            output = matching(data)
            pre_clf = []
            for c in classifiers:
                p = c(output)
                p = nn.Softmax(1)(p)
                pre_clf.append(p)

            pre = pre_clf[0]/3 + pre_clf[1]/3 + pre_clf[2]/3
            pre = torch.max(pre.data, 1)
            n_correct1 += (pre[1] == label).sum().item()
        acc = float(n_correct1) / len(testloader.dataset) * 100
    return acc

def evaluate_source(matching, classifier, loaderlist, device):
    matching.eval()
    classifier.eval()
    n_correct = 0
    num = 0
    loaderlist = [loaderlist]
    with torch.no_grad():
        for testloader in loaderlist:
            for i, (data, label) in enumerate(testloader):
                data, label = data.to(device), label.to(device)
                output = matching(data)
                
                output= classifier(output)
                pre = nn.Softmax()(output)
                pre = torch.max(pre.data, 1)
                n_correct += (pre[1] == label).sum().item()
            num += len(testloader.dataset)
    acc = float(n_correct) / num * 100
    return acc


