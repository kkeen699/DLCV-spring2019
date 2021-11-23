import torch
import torch.nn as nn
import torch.nn.functional as F

class YoloLoss(nn.Module):

    def __init__(self, l_coord, l_noobj):
        super(YoloLoss, self).__init__()
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def forward(self, predict, label):
        N = predict.size()[0]

        coord_mask = label[:, :, :, 4] > 0
        noobj_mask = label[:, :, :, 4] == 0
        coord_mask = coord_mask.unsqueeze(-1).expand_as(label)
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(label)
        

        coord_pre = predict[coord_mask].view(-1, 26)
        noobj_pre = predict[noobj_mask].view(-1, 26)
        coord_lab = label[coord_mask].view(-1, 26)

        # no object loss 
        noobj_pre_c = torch.cat((noobj_pre[:,4].view(-1,1), noobj_pre[:,9].view(-1,1)), 1)
        noobj_lab_c = torch.zeros(noobj_pre_c.size()).to(device)
        noobj_c_loss = F.mse_loss(noobj_pre_c, noobj_lab_c, reduction='sum')

        # contain object loss
        class_pre = coord_pre[:, 10:]
        box_pre = coord_pre[:, :10]
        class_lab = coord_lab[:, 10:]
        box_lab = coord_lab[:, :5]

        class_loss = F.mse_loss(class_pre, class_lab, reduction='sum')
        
        obj_c_loss = 0.
        coord_loss = 0.
        for i in range(box_pre.size()[0]):
            if IoU(box_pre[i][:4], box_lab[i][:4]) > IoU(box_pre[i][5:9], box_lab[i][:4]):
                coord_loss += (F.mse_loss(box_pre[i][:2], box_lab[i][:2], reduction='sum') 
                             + F.mse_loss(torch.sqrt(box_pre[i][2:4]), torch.sqrt(box_lab[i][2:4]), reduction='sum'))
                obj_c_loss += F.mse_loss(box_pre[i][4], IoU(box_pre[i][:4], box_lab[i][:4]), reduction='sum')
                noobj_c_loss += F.mse_loss(box_pre[i][9], torch.zeros(1).to(device), reduction='sum')
            else:
                coord_loss += (F.mse_loss(box_pre[i][5:7], box_lab[i][:2], reduction='sum') 
                             + F.mse_loss(torch.sqrt(box_pre[i][7:9]), torch.sqrt(box_lab[i][2:4]), reduction='sum'))
                obj_c_loss += F.mse_loss(box_pre[i][9], IoU(box_pre[i][5:9], box_lab[i][:4]), reduction='sum')
                noobj_c_loss += F.mse_loss(box_pre[i][4], torch.zeros(1).to(device), reduction='sum')

        return (self.l_coord*coord_loss + obj_c_loss + self.l_noobj*noobj_c_loss + class_loss)/N

def IoU(box1, box2):

    b1 = [box1[0]/7 - box1[2]/2, box1[1]/7 - box1[3]/2, box1[0]/7 + box1[2]/2, box1[1]/7 + box1[3]/2]
    b2 = [box2[0]/7 - box2[2]/2, box2[1]/7 - box2[3]/2, box2[0]/7 + box2[2]/2, box2[1]/7 + box2[3]/2]
    S1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    S2 = (b2[2] - b2[0]) * (b2[3] - b2[1])

    left = max(b1[0], b2[0])
    right = min(b1[2], b2[2])
    top = max(b1[1], b2[1])
    bottom = min(b1[3], b2[3])
    if left >= right or top >= bottom:
        return torch.tensor(0.).to(device)
    else:
        intersection = (right - left)*(bottom - top)
        return (intersection / (S1 + S2 - intersection)).to(device)
    

