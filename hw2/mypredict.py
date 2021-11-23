import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from vgg_yolo import Yolov1_vgg16bn
import vali
from dataset import yoloDataset

classes = ['plane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court', 'basketball-court',
           'ground-track-field', 'harbor', 'bridge', 'small-vehicle', 'large-vehicle', 'helicopter',
           'roundabout', 'soccer-ball-field', 'swimming-pool', 'container-crane']

def validation(model):
    
    model.eval()
    iteration = 0
    with torch.no_grad():
        for data, target in testset_loader:
            data, target = data.to(device), target.to(device)
            output = model(data).view(7, 7, 26)
            box = NMS(output)
            output_file(box, iteration)
            iteration += 1
    
    
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
    
def NMS(predict):
    
    score_threshold = nn.Threshold(0.1, 0)
    choice = torch.Tensor().to(device)
  
    c = torch.stack((predict[:, :, 4], predict[:, :, 9]), 2)
    max_class, index = torch.max(predict[:, :, 10:], 2)
    max_class = max_class.view(7, 7, 1).expand_as(c)
    index = index.type(torch.float)
    score = score_threshold(c * max_class)

    candidate = torch.zeros(1, 6).to(device)
    for i in range(7):
        for j in range(7):
            for k in range(2):
                if score[i, j, k] > 0:
                    box = torch.add(predict[i, j, 5*k:5*k+4], torch.tensor([j, i, 0., 0.]).to(device))
                    box = torch.cat((box, index[i,j].view(1), score[i,j,k].view(1)), 0)
                    box = box.view(1,-1)
                    candidate = torch.cat((candidate, box), 0) 
    
    score = candidate[:, -1]
    while torch.max(score).item() :
        idx = score.argmax()
        choice = torch.cat((choice, candidate[idx].view(1, -1)), 0)
        box_c = candidate[idx, :4]

        candidate[idx, -1] = 0.
        score[idx] = 0. 
        for i in range(candidate.size()[0]):
            if candidate[i, -1]:
                if IoU(box_c, candidate[i, :4]) > 0.4:
                    candidate[i, -1] = 0.
                    score[i] = 0.
    return choice

                                  
                                  
def output_file(box, index):
    box = box.to('cpu')
    with open('./prediction_0417/' + str(index).zfill(4) + ".txt", 'w') as f:
        for a in box:
            x = a[0].item() * 512 / 7
            y = a[1].item() * 512 / 7
            w = a[2].item() * 512
            h = a[3].item() * 512
            s = a[5].item()
            xmin = round(x - w/2, 2)
            ymin = round(y - h/2, 2)
            xmax = round(x + w/2, 2)
            ymax = round(y + h/2, 2)
            f.write(str(xmin) + ' ' + str(ymin) + ' ' + str(xmax) + ' ' + str(ymin) + ' ' + str(xmax) + ' ' 
                  + str(ymax) + ' ' + str(xmin) + ' ' + str(ymax) + ' ' + classes[int(a[4].item())] + ' '
                  + str(round(s, 2)) + '\n')
    

if __name__ == '__main__':
    
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else "cpu")
    print('Device used:', device)

    model = Yolov1_vgg16bn(pretrained=True).to(device)

    file_root = '/home/data/dlcv_datasets/hw2_train_val/val1500/'
    testset = yoloDataset(root = file_root, num = 1500, train =False, transform = transforms.ToTensor())
    testset_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)
    
    for i in range(4,100, 5):
        print('ep = ' + str(i) + ' ==========================')
        checkpoint_path = './model_0417/mnist-' + str(i) + '.pth'
        state = torch.load(checkpoint_path)
        model.load_state_dict(state['state_dict'])
        print('model loaded from %s' % checkpoint_path)
        validation(model)
        vali.val('./prediction_0417', '/home/data/dlcv_datasets/hw2_train_val/val1500/labelTxt_hbb')
   
    
        
    
    
