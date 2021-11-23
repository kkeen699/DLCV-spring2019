import pandas as pd
from os import listdir

label_file = './hw4_data/TrimmedVideos/label/gt_valid.csv'

df = pd.read_csv(label_file)
action_labels = df['Action_labels'].tolist()

with open('/home/kkeen/Documents/hw4_pre/p1_valid.txt','r') as f:
    p1 = [int(w.strip()) for w in f.readlines()]

with open('/home/kkeen/Documents/hw4_pre/p2_result.txt','r') as f:
    p2 = [int(w.strip()) for w in f.readlines()]

acc1 = 0
acc2 = 0
l = len(action_labels)

for i in range(l):
    if action_labels[i] == p1[i]:
        acc1 += 1
    if action_labels[i] == p2[i]:
        acc2 += 1

acc1 = acc1 / l * 100
acc2 = acc2 / l * 100
print('p1 acc:', acc1, 'p2 acc:', acc2)


video_path = './hw4_data/FullLengthVideos/labels/valid'
category_list = sorted(listdir(video_path))

p3_correct = 0
p3_len = 0
for category in category_list:
    n_correct = 0
    with open(video_path+'/'+category,'r') as f:
        label = [int(w.strip()) for w in f.readlines()]
    with open('/home/kkeen/Documents/hw4_pre/'+category, 'r') as f:
        pre = [int(w.strip()) for w in f.readlines()]
    for i in range(len(label)):
        if label[i] == pre[i]:
            n_correct += 1
    c_acc = n_correct / len(label) * 100
    print(category, c_acc)
    p3_len += len(label)
    p3_correct += n_correct
acc3 = p3_correct / p3_len * 100
print('p3 acc:', acc3)
