import sys
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable

from yoloLoss import yoloLoss
from dataset import yoloDataset

import numpy as np
from resnet_yolo import resnet50
from vgg_yolo import Yolov1_vgg16bn

def save_checkpoint(checkpoint_path, model, optimizer):
    
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

m = sys.argv[1]
use_gpu = torch.cuda.is_available()

file_root = '/home/data/dlcv_datasets/hw2_train_val/train15000/'
test_root = '/home/data/dlcv_datasets/hw2_train_val/val1500/'
learning_rate = 0.001
num_epochs = 100
batch_size = 8

if m == 'vgg16':
    net = Yolov1_vgg16bn(pretrained=True)
elif m == 'resnet':
    net = resnet50(pretrained=True)
else:
    print('wrong model')

criterion = yoloLoss(7,2,5,0.5)

if use_gpu:
    print('cuda')
    net.cuda()

net.train()

params=[]
params_dict = dict(net.named_parameters())
for key,value in params_dict.items():
    if key.startswith('features'):
        params += [{'params':[value],'lr':learning_rate*1}]
    else:
        params += [{'params':[value],'lr':learning_rate}]
optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)

train_dataset = yoloDataset(root=file_root, num = 15000, train=True, transform = transforms.ToTensor() )
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=1)

test_dataset = yoloDataset(root=test_root, num = 1500,train=False,transform = transforms.ToTensor() )
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=1)


num_iter = 0

best_test_loss = np.inf

for epoch in range(num_epochs):
    net.train()
    
    if epoch == 30:
        learning_rate=0.0001
    if epoch == 40:
        learning_rate=0.00001

    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    
    print('\n\nStarting epoch %d / %d' % (epoch + 1, num_epochs))
    print('Learning Rate for this epoch: {}'.format(learning_rate))
    
    total_loss = 0.
    
    for i,(images,target) in enumerate(train_loader):
        images = Variable(images)
        target = Variable(target)
        if use_gpu:
            images,target = images.cuda(),target.cuda()
        
        pred = net(images)
        loss = criterion(pred,target)
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 150 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f' 
            %(epoch+1, num_epochs, i+1, len(train_loader), loss.item(), total_loss / (i+1)))
            num_iter += 1

    #validation
    validation_loss = 0.0
    net.eval()
    for i,(images,target) in enumerate(test_loader):
        images = Variable(images,volatile=True)
        target = Variable(target,volatile=True)
        if use_gpu:
            images,target = images.cuda(),target.cuda()
        
        pred = net(images)
        loss = criterion(pred,target)
        validation_loss += loss.item()
    validation_loss /= len(test_loader)
    print('vali loss = ', validation_loss, '\n')
    
    save_checkpoint('./model_0417/mnist-%i.pth' % epoch, net, optimizer)
    

    
    
    
    
    
    
    
    