import torch
import torchvision
import torchvision.transforms as transforms
import os
from os import listdir
import cv2
import pickle
from p3_model import ResNet_feature_extractor

def normalize(image):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_input = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad((0,40), fill=0, padding_mode='constant'),
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize
        ])
    return transform_input(image)

def cut_frames(features_per_category, labels_per_category, size = 200, overlap = 20):
    a = torch.split(features_per_category, size-overlap)
    b = torch.split(torch.Tensor(labels_per_category), size-overlap)

    cut_features = []
    cut_labels = []
    for i in range(len(a)):
        if i==0:
            cut_features.append(a[i])
            cut_labels.append(b[i])
        else:
            cut_features.append(torch.cat((a[i-1][-overlap:],a[i])))
            cut_labels.append(torch.cat((b[i-1][-overlap:],b[i])))
    
    lengths = [len(f) for f in cut_labels]

    return cut_features, cut_labels, lengths



use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print('Device used', device)

feature_extractor = ResNet_feature_extractor().to(device)
feature_extractor.eval()

# train set
video_path = '../hw4_data/FullLengthVideos/videos/train/'
category_list = sorted(listdir(video_path))

with torch.no_grad():
    train_all_video_frame = []
    for category in category_list:
        image_list_per_folder = sorted(listdir(os.path.join(video_path,category)))
        category_frames = torch.Tensor()
        for image in image_list_per_folder:
            image = cv2.imread(os.path.join(video_path, category,image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = normalize(image)
            image = image.unsqueeze(0).to(device)
            feature = feature_extractor(image).cpu()
            category_frames = torch.cat((category_frames, feature), 0)
        train_all_video_frame.append(category_frames)


# train label
label_path = '../hw4_data/FullLengthVideos/labels/train/'
category_txt_list = sorted(listdir(label_path))
train_category_labels = []
for txt in category_txt_list:
    file_path = os.path.join(label_path,txt)
    with open(file_path,'r') as f:
        label = [int(w.strip()) for w in f.readlines()]
        train_category_labels.append(label)

cutting_steps = 350
overlap_steps = 30
train_cut_features = []
train_cut_labels = []
train_cut_lengths = []
for category_frames, category_labels in zip(train_all_video_frame,train_category_labels):
    features, labels, lengths = cut_frames(category_frames,category_labels, size = cutting_steps, overlap = overlap_steps)
    train_cut_features += features
    train_cut_labels += labels
    train_cut_lengths += lengths

with open('./features/train_features.pickle', 'wb') as f:
    pickle.dump(train_cut_features,f)
with open('./features/train_labels.pickle', 'wb') as f:
    pickle.dump(train_cut_labels,f)
with open('./features/train_lengths.pickle', 'wb') as f:
    pickle.dump(train_cut_lengths,f)


# validation set
video_path = '../hw4_data/FullLengthVideos/videos/valid/'
category_list = sorted(listdir(video_path))

with torch.no_grad():
    valid_all_video_frame = []
    for category in category_list:
        image_list_per_folder = sorted(listdir(os.path.join(video_path,category)))
        category_frames = torch.Tensor()
        for image in image_list_per_folder:
            image = cv2.imread(os.path.join(video_path, category,image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = normalize(image)
            image = image.unsqueeze(0).to(device)
            feature = feature_extractor(image).cpu()
            category_frames = torch.cat((category_frames, feature), 0)
        valid_all_video_frame.append(category_frames)

# valid label
label_path = '../hw4_data/FullLengthVideos/labels/valid/'
category_txt_list = sorted(listdir(label_path))
valid_category_labels = []
for txt in category_txt_list:
    file_path = os.path.join(label_path,txt)
    with open(file_path,'r') as f:
        label = [int(w.strip()) for w in f.readlines()]
        valid_category_labels.append(label)
        
valid_lengths = [len(s) for s in valid_all_video_frame]

with open('./features/valid_features.pickle', 'wb') as f:
    pickle.dump(valid_all_video_frame,f)
with open('./features/valid_labels.pickle', 'wb') as f:
    pickle.dump(valid_category_labels,f)
with open('./features/valid_lengths.pickle', 'wb') as f:
    pickle.dump(valid_lengths,f)
