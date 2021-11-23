import torch
import pickle
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


with open('/home/kkeen/Documents/RNN_features.pickle', 'rb') as f:
    RNN_feature = pickle.load(f)
with open('/home/kkeen/Documents/valid_label.pickle', 'rb') as f:
    label = pickle.load(f)

feature = np.empty(shape=[0, 512])
for f in RNN_feature:
    f = f.numpy()
    f = f.reshape(1, -1)
    feature = np.append(feature, f, axis=0)

label = np.array(label)


CNN_features_2d = TSNE(n_components=2, random_state=0).fit_transform(feature)
cm = plt.cm.get_cmap('tab20', 11)
plt.figure(figsize=(10,5))
plt.scatter(CNN_features_2d[:,0], CNN_features_2d[:,1], c=label, cmap=cm)
plt.colorbar(ticks=range(11))
plt.clim(-0.5, 10.5)
plt.savefig('p2_tsne.png')
plt.show()
