import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np

classes = ['plane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court', 'basketball-court',
           'ground-track-field', 'harbor', 'bridge', 'small-vehicle', 'large-vehicle', 'helicopter',
           'roundabout', 'soccer-ball-field', 'swimming-pool', 'container-crane']
class yoloDataset(data.Dataset):
   
    def __init__(self, root, num, train,transform):
        self.root= root
        self.num = num
        self.train = train
        self.transform = transform
        self.imagenames = []
        self.labelnames = []
        self.image_size = 448
        
        if train:
            for i in range(num):
                self.imagenames.append(root + 'images/' + str(i).zfill(5) + '.jpg')
                self.labelnames.append(root + 'labelTxt_hbb/' + str(i).zfill(5) + '.txt')
        else:
            for i in range(num):
                self.imagenames.append(root + 'images/' + str(i).zfill(4) + '.jpg')
                self.labelnames.append(root + 'labelTxt_hbb/' + str(i).zfill(4) + '.txt')

    def __getitem__(self,idx):
        image = cv2.imread(self.imagenames[idx])
        H = image.shape[0]
        W = image.shape[1]
        
        with open(self.labelnames[idx]) as f:
            lines = f.readlines()
        boxes = []
        for line in lines:
            splited = line.strip().split()
            x = (float(splited[2]) + float(splited[0]))/2
            y = (float(splited[5]) + float(splited[1]))/2
            w = float(splited[2]) - float(splited[0])
            h = float(splited[5]) - float(splited[1])
            c = np.zeros(16)
            c[classes.index(splited[8])] = 1
            boxes.append(np.append([x, y, w, h], c))

        grid_num = 7
        label = torch.zeros(grid_num, grid_num, 26)
        cell_size = 1./grid_num
        
        for obj in boxes:
            i = int((obj[0] / W) // cell_size)
            j = int((obj[1] / H) // cell_size) 
            
            if label[j, i, 4] == 0:
                label[j, i, 0] = (obj[0] / W) / cell_size - i
                label[j, i, 1] = (obj[1] / H) / cell_size - j
                label[j, i, 2] = obj[2] / W
                label[j, i, 3] = obj[3] / H
                label[j, i, 4] = 1
                label[j, i, 5] = (obj[0] / W) / cell_size - i
                label[j, i, 6] = (obj[1] / H) / cell_size - j
                label[j, i, 7] = obj[2] / W
                label[j, i, 8] = obj[3] / H
                label[j, i, 9] = 1
                label[j, i, 10:] = torch.tensor(obj[4:])
                
        image = self.BGR2RGB(image)
        image = cv2.resize(image, (self.image_size, self.image_size))

        if self.transform is not None:
            image = self.transform(image).type(torch.float)
        return image, label
    
    def __len__(self):
        return self.num

    def BGR2RGB(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


