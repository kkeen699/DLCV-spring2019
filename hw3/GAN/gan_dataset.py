import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import glob
import cv2
import os

class GanDataset(data.Dataset):
    def __init__(self, root):
        self. filename = sorted(glob.glob(os.path.join(root,'*.png')))

    def __getitem__(self, index):
        image = cv2.imread(self.filename[index])
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = image/255
        image = transforms.ToTensor()(image).type(torch.float)
        label = torch.ones(1)
        return image, label

    def __len__(self):
        return len(self.filename)
