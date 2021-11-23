import torch
import torchvision.transforms as transforms
import torch.nn as nn

import models
from resnet_yolo import resnet50

import numpy as np
import glob
import sys
import cv2
import os

classes = ['plane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court', 'basketball-court',
           'ground-track-field', 'harbor', 'bridge', 'small-vehicle', 'large-vehicle', 'helicopter',
           'roundabout', 'soccer-ball-field', 'swimming-pool', 'container-crane']

def decoder(pred):

    grid_num = 7
    boxes=[]
    cls_indexs=[]
    probs = []
    cell_size = 1./grid_num
    pred = pred.data
    pred = pred.squeeze(0)
    contain1 = pred[:,:,4].unsqueeze(2)
    contain2 = pred[:,:,9].unsqueeze(2)
    contain = torch.cat((contain1,contain2),2)
    mask1 = contain > 0.1 
    mask2 = (contain==contain.max())
    mask = (mask1+mask2).gt(0)
    
    for i in range(grid_num):
        for j in range(grid_num):
            for b in range(2):
                if mask[i,j,b] == 1:
                    box = pred[i,j,b*5:b*5+4]
                    contain_prob = torch.FloatTensor([pred[i,j,b*5+4]])
                    xy = torch.FloatTensor([j,i])*cell_size 
                    box[:2] = box[:2]*cell_size + xy
                    box_xy = torch.FloatTensor(box.size())
                    box_xy[:2] = box[:2] - 0.5*box[2:]
                    box_xy[2:] = box[:2] + 0.5*box[2:]
                    max_prob,cls_index = torch.max(pred[i,j,10:],0)
                    if float((contain_prob*max_prob)[0]) > 0.1:
                        boxes.append(box_xy.view(1,4))
                        cls_indexs.append(cls_index.view(1))
                        probs.append(contain_prob*max_prob)
    
    if len(boxes) ==0:
        boxes = torch.zeros((1,4))
        probs = torch.zeros(1)
        cls_indexs = torch.zeros(1)
    else:
        boxes = torch.cat(boxes,0)
        probs = torch.cat(probs,0)
        cls_indexs = torch.cat(cls_indexs,0)
    keep = nms(boxes,probs)
    return boxes[keep],cls_indexs[keep],probs[keep]

def nms(bboxes,scores,threshold=0.5):
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    areas = (x2-x1) * (y2-y1)
    
    _,order = scores.sort(0,descending=True)
    
    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            i = order
            keep.append(i)
            break
            
        i = order[0]
        keep.append(i)

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2-xx1).clamp(min=0)
        h = (yy2-yy1).clamp(min=0)
        inter = w*h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (ovr<=threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1]
    return torch.LongTensor(keep)
            
def output_file(boxes, cls_indexs, probs, index, root, name):
    with open(root + '/' + name + ".txt", 'w') as f:
        for i,box in enumerate(boxes):
            xmin = round(box[0].item()*512, 2)
            ymin = round(box[1].item()*512, 2)
            xmax = round(box[2].item()*512, 2)
            ymax = round(box[3].item()*512, 2)
            cls_index = int(cls_indexs[i])
            prob = float(probs[i])
            f.write(str(xmin) + ' ' + str(ymin) + ' ' + str(xmax) + ' ' + str(ymin) + ' ' + str(xmax) + ' ' 
                  + str(ymax) + ' ' + str(xmin) + ' ' + str(ymax) + ' ' + classes[cls_index] + ' '
                  + str(prob) + '\n')



if __name__ == '__main__':
    modelname = sys.argv[1]
    image_root = sys.argv[2]
    output_root = sys.argv[3]

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('Device used:', device)

    if modelname == 'vgg16':
        model = models.Yolov1_vgg16bn(pretrained=True).to(device)
        checkpoint_path = './vgg16_yolo_ep50.pth'
    elif modelname == 'resnet':
        model = resnet50().to(device)
        checkpoint_path = './resnet_yolo_ep20.pth'
    else:
        print('wrong model')

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state['state_dict'])
    model.eval()
    
    filename = sorted(glob.glob(os.path.join(image_root,'' ,'*.jpg')))

    for i in range(len(filename)):
        img = cv2.cvtColor(cv2.imread(filename[i]), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(448,448))
        img = transforms.ToTensor()(img)
        img = img[None,:,:,:]
        img = img.to(device)

        name = filename[i].split('/')
        name = name[-1][:-4]

        output = model(img)
        output = output.cpu()
        boxes,cls_indexs,probs =  decoder(output)
        output_file(boxes, cls_indexs, probs, i, output_root, name)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
