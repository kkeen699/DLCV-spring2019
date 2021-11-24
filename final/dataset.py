import torch.utils.data as data
import torchvision.transforms as transforms
import pandas as pd
import cv2
import os
import pickle
import glob

def normalize(image):
    '''
    normalize for pre-trained model input
    '''
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    x = image.shape[1]
    y = image.shape[0]
    
    if x > y:
        s = int((x-y)/2)
        transform_input = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Pad((0,s), fill=0, padding_mode='constant'),
                #transforms.Resize(224),
                transforms.ToTensor(),
                normalize
            ])
    else:
        s = int((y-x)/2)
        transform_input = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Pad((s,0), fill=0, padding_mode='constant'),
                #transforms.Resize(224),
                transforms.ToTensor(),
                normalize
            ])
    return transform_input(image)


class M3SDA_Dataset(data.Dataset):
    def __init__(self, root, domain, train):
        if train:
            file = os.path.join(root, domain, domain+'_train.csv')
        else:
            file = os.path.join(root, domain, domain+'_test.csv')
        self.root = root
        df = pd.read_csv(file)
        self.image_name = df['image_name'].tolist()
        self.label = df['label'].tolist()
    
    def __getitem__(self, idx):
        path = os.path.join(self.root, self.image_name[idx])
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224,224), interpolation=cv2.INTER_CUBIC)
        image = normalize(image)
        label = int(self.label[idx])
        return image, label

    def __len__(self):
        return len(self.image_name)


class DataFeature(data.Dataset):
    def __init__(self, domain, train):

        if domain == 'real' and train == 'test':
            self.f = False
        else:
            self.f = True
        
        feature_file = './feature/'+domain+'_'+train+'_feature.pickle'
        with open(feature_file, 'rb') as f:
            self.features = pickle.load(f) 
        
        if self.f:
            label_file = './feature/'+domain+'_'+train+'_label.pickle'
            with open(label_file, 'rb') as f:
                self.labels = pickle.load(f)

    def __getitem__(self, idx):
        feature = self.features[idx]

        if self.f:
            label = self.labels[idx]
            return feature, label
        else:
            return feature

    def __len__(self):
        return len(self.features)

class RealDataset(data.Dataset):
    def __init__(self):
        root = '/home/kkeen/dlcv/final_data/test'
        self.filename = sorted(glob.glob(os.path.join(root,'*.jpg')))
    
    def __getitem__(self, idx):
        image = cv2.imread(self.filename[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224,224), interpolation=cv2.INTER_CUBIC)
        image = normalize(image)
        return image

    def __len__(self):
        return len(self.filename)
