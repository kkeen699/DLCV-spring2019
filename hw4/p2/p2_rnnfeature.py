import torch
import torch.nn as nn
import numpy as np
import pickle
from p2_model import LSTM


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

with open('./features/valid_feature.pickle', 'rb') as f:
    valid_feature = pickle.load(f)
with open('./features/valid_label.pickle', 'rb') as f:
    valid_label = pickle.load(f)

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print('Device used', device)

lstm = LSTM().to(device)
lstm.load_state_dict(torch.load('./model/lstm_epoch_100.pth'))

lstm.eval()
RNN_feature = []
n_correct = 0
with torch.no_grad():
    for i in range(len(valid_label)):
        valid_X, valid_y, length = single_batch_padding([valid_feature[i]], [valid_label[i]], test=True)
        valid_X, valid_y = valid_X.to(device), valid_y.to(device)

        output, RNN_hidden = lstm(valid_X, length)
        print(RNN_hidden.size())
        pred = torch.max(output.data, 1)
        n_correct += (pred[1] == valid_y).sum().item()
        RNN_feature.append(RNN_hidden.cpu().data.squeeze())
    acc = float(n_correct) / len(valid_label) * 100
print('Validation accuracy:', acc)
print(RNN_feature[0].size())
with open('./features/RNN_features.pickle', 'wb') as f:
    pickle.dump(RNN_feature, f)
