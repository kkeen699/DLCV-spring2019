import torch
import torch.nn as nn
import torch.optim as optim
import sys
from model import Classifier, MomentMatching
from utils import getdata, evaluate, MD2_2


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print('Device used', device)

target = sys.argv[1]
BATCH_SIZE = 1024
source_trainloader, target_trainloader, target_testloader = getdata(target, BATCH_SIZE)
source1_trainloader, source2_trainloader, source3_trainloader = source_trainloader[0], source_trainloader[1], source_trainloader[2]

matching = MomentMatching().to(device)
classifiers = []
for i in range(3):
    classifiers.append(Classifier().to(device))

parameter = list(matching.parameters())
for c in classifiers:
    parameter += list(c.parameters())
optimizer = optim.Adam(parameter, lr=1e-3, weight_decay=1e-4)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

class_criterion = nn.CrossEntropyLoss()
#domiain_criterion = MD2()
domiain_criterion = MD2_2()
#domiain_criterion = CORAL_loss()


for epoch in range(500):
    print('\nepoch', epoch+1, '====================================')
    scheduler.step()
    matching.train()
    for c in classifiers:
        c.train()
    epo_loss = 0
    epo_domain = 0
    epo_class = 0
    dataset = zip(source1_trainloader, source2_trainloader, source3_trainloader, target_trainloader)
    
    for i, ((s1_data, s1_label), (s2_data, s2_label), (s3_data, s3_label), (t_data, t_label)) in enumerate(dataset):
        s1_data, s1_label = s1_data.to(device), s1_label.to(device)
        s2_data, s2_label = s2_data.to(device), s2_label.to(device)
        s3_data, s3_label = s3_data.to(device), s3_label.to(device)
        t_data, t_label = t_data.to(device), t_label.to(device)

        source_data = [s1_data, s2_data, s3_data]
        source_label = [s1_label, s2_label, s3_label]

        optimizer.zero_grad()

        # moment matching
        source_feature = []
        for s in source_data:
            feature = matching(s)
            source_feature.append(feature)
        target_feature = matching(t_data)
        
        loss_domain = domiain_criterion(source_feature, target_feature)

        # classifier
        pre = []
        loss_class = 0.0
        for feature, clf, label in zip(source_feature, classifiers, source_label):
            output = clf(feature)
            loss_class += class_criterion(output, label)

        loss = loss_domain + loss_class

        loss.backward()
        optimizer.step()
        epo_loss += loss.item()
        epo_domain += loss_domain.item()
        epo_class += loss_class.item()
    print('Loss:', epo_loss)
    print('domian loss:', epo_domain, 'class loss:', epo_class)

    # validation
    if target != 'real':
        acc = evaluate(matching, classifiers, target_testloader, device)
        print('valadation accuracy:', acc)
'''
    if (epoch+1) % 10 == 0:
        torch.save(matching.state_dict(), './model/'+ target +'/matching_%d.pth' % (epoch+1))
        for i, c in enumerate(classifiers):
            torch.save(c.state_dict(), './model/' + target + '/clf%d_%d.pth' % (i, epoch+1))
'''





    
