import torch
import torch.nn as nn
import numpy as np
import pickle
from p2_model import LSTM
import matplotlib.pyplot as plt


def single_batch_padding(train_X_batch, train_y_batch, test = False):
    if test==True:
        padded_sequence = nn.utils.rnn.pad_sequence(train_X_batch)
        label = torch.LongTensor(train_y_batch)
        length = [len(train_X_batch[0])]
    else:
        length = [len(x) for x in train_X_batch]
        perm_index = np.argsort(length)[::-1]

        # sort by sequence length
        train_X_batch = [train_X_batch[i] for i in perm_index]
        length = [len(x) for x in train_X_batch]
        padded_sequence = nn.utils.rnn.pad_sequence(train_X_batch)
        label = torch.LongTensor(np.array(train_y_batch)[perm_index])
    return padded_sequence, label, length


with open('./features/train_feature.pickle', 'rb') as f:
    train_feature = pickle.load(f)
with open('./features/valid_feature.pickle', 'rb') as f:
    valid_feature = pickle.load(f)
    
with open('./features/train_label.pickle', 'rb') as f:
    train_label = pickle.load(f)
with open('./features/valid_label.pickle', 'rb') as f:
    valid_label = pickle.load(f)

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print('Device used', device)

lstm = LSTM().to(device)
optimizer = torch.optim.Adam(lstm.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()
BATCH_SIZE = 64

train_loss = []
validation_acc = []
RNN_feature = []

for epoch in range(100):
    print('\nepoch', epoch+1, '================================')
    lstm.train()
    total_length = len(train_feature)
    # shuffle
    perm_index = np.random.permutation(len(train_feature))
    train_X_sfl = [train_feature[i] for i in perm_index]
    train_y_sfl = np.array(train_label)[perm_index]

    epo_loss = 0

    for index in range(0, total_length, BATCH_SIZE):
        if index + BATCH_SIZE > total_length:
            break

        optimizer.zero_grad()
        input_X = train_X_sfl[index:index+BATCH_SIZE]
        input_y = train_y_sfl[index:index+BATCH_SIZE]

        # pad the data
        input_X, input_y, length = single_batch_padding(input_X, input_y)
        input_X, input_y = input_X.to(device), input_y.to(device)

        output, _ = lstm(input_X, length)
        loss = criterion(output, input_y)
        loss.backward()
        optimizer.step()
        
        epo_loss += loss.item()
    
    print('Loss:', epo_loss)
    train_loss.append(epo_loss)

    # validation
    lstm.eval()
    n_correct = 0
    with torch.no_grad():
        for i in range(len(valid_label)):
            valid_X, valid_y, length = single_batch_padding([valid_feature[i]], [valid_label[i]], test=True)
            valid_X, valid_y = valid_X.to(device), valid_y.to(device)

            output, RNN_hidden = lstm(valid_X, length)
            pred = torch.max(output.data, 1)
            n_correct += (pred[1] == valid_y).sum().item()
            RNN_feature.append(RNN_hidden.cpu().data.squeeze())
        acc = float(n_correct) / len(valid_label) * 100
        validation_acc.append(acc)
    print('Validation accuracy:', acc)

    if (epoch+1) % 5 == 0:
        torch.save(lstm.state_dict(), './model/lstm_epoch_%d.pth' % (epoch+1))

with open('./features/RNN_feature.pickle', 'wb') as f:
    pickle.dump(RNN_feature, f)

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
plt.savefig('p2_curve.png')
plt.show()
