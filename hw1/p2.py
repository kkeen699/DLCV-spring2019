import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# read data
train_set = [str(i)+"_"+str(j)+".png" for i in range(1, 41) for j in range(1, 7)]
test_set = [str(i)+"_"+str(j)+".png" for i in range(1, 41) for j in range(7, 11)]

xtrain = [cv2.imread("./dateset/p2_data/" + n, 0).reshape(-1) for n in train_set]
ytrain = [i for i in range(1, 41) for j in range(1, 7)]
xtrain = np.array(xtrain)
ytrain = np.array(ytrain)

xtest = [cv2.imread("./dateset/p2_data/" + n, 0).reshape(-1) for n in test_set]
ytest = [i for i in range(1, 41) for j in range(7, 11)]
xtest = np.array(xtest)
ytest = np.array(ytest)

# mean face and first four eigenfaces
mean_face = (xtrain.mean(axis = 0)).reshape(1, -1)

plt.subplot(151)
plt.title("mean face")
plt.imshow(mean_face.reshape(56, 46), cmap = "gray")
plt.imsave("mean face.png", mean_face.reshape(56, 46), cmap= "gray")


pca = PCA(n_components = len(xtrain) - 1)
ei = pca.fit(xtrain - mean_face)

for i in range(4):
    plt.subplot(152 + i)
    plt.title("eigenface " + str(i+1))
    plt.imshow(ei.components_[i].reshape(56, 46), cmap = "gray")
plt.show()

# reconstruct image using the first n = 3, 45, 140, 229 eigenfaces
n = [3, 45, 140, 229]
origin = xtrain[0].reshape(1, -1)
a = pca.transform(origin - mean_face)

plt.subplot(151)
plt.title("original image")
plt.imshow(origin.reshape(56, 46), cmap = "gray")

for i in range(4):
    img = mean_face + (a[:, :n[i]] @ pca.components_[:n[i]])
    mse = np.mean((origin - img) ** 2)
    plt.subplot(152 + i)
    plt.title("n = " + str(n[i]) + ", mse = " + str(np.round(mse, 4)))
    plt.imshow(img.reshape(56, 46), cmap = "gray")
plt.show()

# classifier
n = [3, 45, 140] 
xtrain_reduction = pca.transform(xtrain - mean_face)
param_grid = {"n_neighbors" : [1, 3, 5]}
knn = KNeighborsClassifier()
clf = GridSearchCV(knn, param_grid, cv = 3)

acc = {}
for i in n:
    clf.fit(xtrain_reduction[:, :i], ytrain)
    acc["n = " + str(i)] = clf.cv_results_["mean_test_score"]
acc = pd.DataFrame.from_dict(acc, orient = 'index')
acc.columns = ["k = 1", "k = 3", "k = 5"]
print(acc)

# choose k = 1, n = 45
xtest_reduction = pca.transform(xtest - mean_face)
knn.n_neighbors = 1
knn.fit(xtrain_reduction[:, :45], ytrain)
ypre = knn.predict(xtest_reduction[:, :45])
print("The recognition rate of the testing set is", accuracy_score(ytest, ypre))
