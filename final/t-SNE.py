import pickle
import numpy as np
import random
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


with open('./feature/sketch/sketch.pickle', 'rb') as f:
    target_feature = pickle.load(f)
with open('./feature/sketch/infograph.pickle', 'rb') as f:
    s1_feature = pickle.load(f)
with open('./feature/sketch/quickdraw.pickle', 'rb') as f:
    s2_feature = pickle.load(f)
with open('./feature/sketch/real.pickle', 'rb') as f:
    s3_feature = pickle.load(f)
target_feature = random.choices(target_feature, k=3000)
s1_feature = random.choices(s1_feature, k=3000)
s2_feature = random.choices(s2_feature, k=3000)
s3_feature = random.choices(s3_feature, k=3000)
feature = np.empty(shape=[0, 2048])
for f in target_feature:
    f = f.numpy()
    f = f.reshape(1, -1)
    feature = np.append(feature, f, axis=0)
for f in s1_feature:
    f = f.numpy()
    f = f.reshape(1, -1)
    feature = np.append(feature, f, axis=0)
for f in s2_feature:
    f = f.numpy()
    f = f.reshape(1, -1)
    feature = np.append(feature, f, axis=0)
for f in s3_feature:
    f = f.numpy()
    f = f.reshape(1, -1)
    feature = np.append(feature, f, axis=0)
    
label = [0]*3000 + [1]*3000 + [2]*3000 + [3]*3000
label = np.array(label)
print(feature.shape, label.shape)


CNN_features_2d = TSNE(n_components=2, random_state=0).fit_transform(feature)
cm = plt.cm.get_cmap('tab20', 4)
plt.figure(figsize=(10,5))
plt.scatter(CNN_features_2d[:,0], CNN_features_2d[:,1], c=label, cmap=cm)
plt.colorbar(ticks=range(11))
plt.clim(-0.5, 10.5)
plt.savefig('sketch.png')
plt.show()


