import torch
import torch.nn as nn
import numpy as np
import pickle
from p3_model import LSTM
import matplotlib.pyplot as plt


def sort_pad(input_feature, input_lengths, input_labels):
    perm_index = np.argsort(input_lengths)[::-1]
    input_feature =  [input_feature[i] for i in perm_index]
    input_labels =  [input_labels[i] for i in perm_index]
    input_lengths = sorted(input_lengths, reverse=True)
    input_feature = nn.utils.rnn.pad_sequence(input_feature, batch_first=True)
    return input_feature, input_labels, input_lengths


with open('./features/train_features.pickle', 'rb') as f:
    train_feature = pickle.load(f)
with open('./features/train_labels.pickle', 'rb') as f:
    train_label = pickle.load(f)
with open('./features/train_lengths.pickle', 'rb') as f:
    train_length = pickle.load(f)
    
with open('./features/valid_features.pickle', 'rb') as f:
    valid_feature = pickle.load(f)
with open('./features/valid_labels.pickle', 'rb') as f:
    valid_label = pickle.load(f)
with open('./features/valid_lengths.pickle', 'rb') as f:
    valid_length = pickle.load(f)
print(train_feature[0].size())
print(valid_feature[0].size())

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print('Device used', device)

lstm = LSTM().to(device)
lstm.load_state_dict(torch.load('../p2/model/lstm_epoch_100.pth'))
optimizer = torch.optim.Adam(lstm.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()
BATCH_SIZE = 32

train_loss = []
validation_acc = []

for epoch in range(300):
    print('\nepoch', epoch+1, '================================')
    lstm.train()
    total_length = len(train_feature)
    # shuffle
    perm_index = np.random.permutation(total_length)
    train_X_sfl = [train_feature[i] for i in perm_index]
    train_y_sfl = [train_label[i] for i in perm_index]
    train_lengths_sfl = np.array(train_length)[perm_index]

    epo_loss = 0

    for index in range(0, total_length, BATCH_SIZE):
        if index + BATCH_SIZE > total_length:
            break

        optimizer.zero_grad()
        input_X = train_X_sfl[index:index+BATCH_SIZE]
        input_y = train_y_sfl[index:index+BATCH_SIZE]
        input_lengths = train_lengths_sfl[index:index+BATCH_SIZE]
        
        input_X, input_y, input_lengths = sort_pad(input_X, input_lengths, input_y)
        input_X = input_X.to(device)
        output = lstm(input_X, input_lengths)

        loss = 0
        for i in range(BATCH_SIZE):
            sample_length = input_lengths[i]
            target = torch.LongTensor(np.array(input_y[i])).to(device)
            pre = output[i][:sample_length]
            l = criterion(pre, target)
            loss += l

        loss.backward()
        optimizer.step()
        
        epo_loss += loss.item()
    
    print('Loss:', epo_loss)
    train_loss.append(epo_loss)

    # validation
    lstm.eval()
    n_correct = 0
    n_label = 0
    with torch.no_grad():
        for valid_X, valid_y, valid_lengths in zip(valid_feature, valid_label, valid_length):
            valid_X = valid_X.unsqueeze(0)
            valid_y = torch.LongTensor(np.array(valid_y))
            valid_X, valid_y = valid_X.to(device), valid_y.to(device)
            #print(valid_X.size())

            output = lstm(valid_X, [valid_lengths]).squeeze()
            pred = torch.max(output.data, 1)
            n_correct += (pred[1] == valid_y).sum().item()
            n_label += len(valid_y)
        acc = float(n_correct) / n_label * 100
        validation_acc.append(acc)
    print('Validation accuracy:', acc)

    if (epoch+1) % 5 == 0:
        torch.save(lstm.state_dict(), './model/s2s_epoch_%d.pth' % (epoch+1))


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.plot(train_loss)
plt.title('training loss')
plt.ylabel('cross entropy')
plt.xlabel('epoch')
plt.subplot(1,2,2)
plt.plot(validation_acc)
plt.title('validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.savefig('p3_curve.png')
plt.show()
