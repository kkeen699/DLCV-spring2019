import numpy as np
import pandas as pd
import random
import cv2
from numpy import linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

classes = ["banana", "fountain", "reef", "tractor"]
xtrain = [cv2.cvtColor(cv2.imread("./dateset/p3_data/" + name + "/" + name + "_" + str(i).zfill(3) + ".JPEG"), cv2.COLOR_BGR2RGB) 
            for name in classes for i in range(1,376)]
xtest = [cv2.cvtColor(cv2.imread("./dateset/p3_data/" + name + "/" + name + "_" + str(i).zfill(3) + ".JPEG"), cv2.COLOR_BGR2RGB) 
            for name in classes for i in range(376, 501)]
xtrain = np.array(xtrain)
xtest = np.array(xtest)
ytrain = np.repeat([0, 1, 2, 3], repeats = 375)
ytest = np.repeat([0, 1, 2, 3], repeats = 125)

def divide(img):
    patches = []
    for n in img:
        r = np.array(np.split(n, 4))
        for i in range(4):
            temp = np.split(r[i], 4, axis=1)
            patches += temp
    patches = np.array(patches)
    return patches

xtrain_patches = divide(xtrain)
xtest_patches = divide(xtest)

# plot patches
n_image = [random.randint(0, 374) for i in range(4)]
n_patched = [random.randint(0, 15) for i in range(3)]
print(n_image, n_patched)
for i in range(4):
    for j in range(3):
        plt.subplot(1, 3, j+1)
        plt.imshow(xtrain_patches[i*6000 + n_image[i]*16 + n_patched[j]])
        plt.axis("off")
    plt.show()

# k-mean and pca
xtrain_patches = xtrain_patches.reshape(len(xtrain_patches), -1)
xtest_patches = xtest_patches.reshape(len(xtest_patches), -1)

kmeans = KMeans(n_clusters=15, max_iter=5000)
kmeans.fit(xtrain_patches)
label = kmeans.labels_
center = kmeans.cluster_centers_

pca = PCA(n_components=3)
pca.fit(xtrain_patches)
xtrain_reduction = pca.transform(xtrain_patches)
center_reduction = pca.transform(center)


sample_list = [0, 1, 2, 3, 4, 5]
color_array = ['b', 'g', 'r', 'c', 'm', 'y']
color_dict = dict()
for i, c in enumerate(sample_list):
    color_dict[c] = color_array[i]
color_list = []
for i in label[np.isin(label, sample_list)]:
    color_list.append(color_dict[i])

sample_center = center_reduction[sample_list]
sample_cluster = xtrain_reduction[np.isin(label, sample_list)]

fig = plt.figure(figsize=(16,8))
ax = Axes3D(fig)
ax.scatter(sample_cluster[:,0], sample_cluster[:,1], sample_cluster[:,2], c=color_list,alpha=0.1)
ax.scatter(sample_center[:,0], sample_center[:,1], sample_center[:,2], s=100, marker = 'D', c=color_array, alpha=1)
ax.set_title("Sample 6 clusters")
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')
plt.show()

# soft-max and plot hitogram of Bow
def soft_max(feature):
    B = []
    for a in feature:
        distance = []
        for f in a:
            temp = [1/LA.norm(f - c) for c in center]
            s = sum(temp)
            temp = [round(i/s,2) for i in temp]
            distance.append(temp)
        distance = np.array(distance)
        B.append(np.amax(distance, axis=0))
    B = np.array(B)
    return B

xtrain_patches = np.array(np.split(xtrain_patches, 1500))
BoW = soft_max(xtrain_patches)

for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.title(classes[i])
    plt.bar(np.arange(BoW.shape[1]), BoW[375*i + n_image[i]])
plt.show()

# knn classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(BoW, ytrain)

xtest_patches = np.array(np.split(xtest_patches, 500))
xtest_BoW = soft_max(xtest_patches)

ypre = knn.predict(xtest_BoW)
print("The recognition rate of the testing set is", accuracy_score(ytest, ypre))
