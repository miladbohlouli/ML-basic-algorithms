import numpy as np
from matplotlib import pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D

def dbscan(input_data, eps, MinPts):
    num_data = input_data.shape[0]
    labels = np.zeros(num_data)
    c = 0

    for index in range(num_data):
        if not (labels[index] == 0):
            continue
        NeighborPts = regionQuery(input_data, index, eps)
        if len(NeighborPts) < MinPts:
            labels[index] = -1
        else:
            c += 1
            labels[index] = c
            i = 0
            while i < len(NeighborPts):
                Pn = NeighborPts[i]
                if labels[Pn] == -1:
                    labels[Pn] = c
                elif labels[Pn] == 0:
                    labels[Pn] = c
                    PnNeighborPts = regionQuery(input_data, Pn, eps)
                    if len(PnNeighborPts) >= MinPts:
                        NeighborPts = np.append(NeighborPts, PnNeighborPts)
                i += 1
    return labels


def regionQuery(D, P, eps):
    return np.where(np.linalg.norm(D - D[P], axis=1) < eps)[0]


def purity(true_labels, pred_labels):
    assert true_labels.shape == pred_labels.shape
    num_data = true_labels.shape[0]
    sum = 0
    for i in np.unique(pred_labels):
        val, counts = np.unique(true_labels[pred_labels == i], return_counts=True)
        if len(counts) != 0:
            sum += np.max(counts)
    return sum / num_data


# path = "datasets/question2"
# file_name = "spiral.txt"
# data = np.genfromtxt(os.path.join(path, file_name))
# eps = 2
# minpts = 3
# label = dbscan(data, eps, minpts)
# print(data.shape)
# plt.suptitle("Clustering the " + file_name + " using DBSCAN algorithm with eps: " + str(eps) + " and minPts: " + str(minpts), fontsize=10)
# plt.subplot(1, 2, 1), plt.title("Original_data with true clustering", fontsize=8)
# for i in np.unique(data[:, 2]):
#     plt.scatter(data[data[:, 2] == i, 0], data[data[:, 2] == i, 1], s=3)
# plt.subplot(1, 2, 2), plt.title("clustered data with purity: %.3f" % purity(data[:, 2], label), fontsize=8)
# for i in np.unique(label):
#     plt.scatter(data[label == i, 0], data[label == i, 1], s=3)
# plt.show()


path = "datasets/question2"
file_name = "rings.txt"
data = np.genfromtxt(os.path.join(path, file_name))
eps = 5
minpts = 3
label = dbscan(data, eps, minpts)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.suptitle("Clustering the " + file_name + " using DBSCAN algorithm with eps: " + str(eps) + " and minPts: " + str(minpts), fontsize=10)
# plt.suptitle("Original_data with true clustering", fontsize=8)
# for i in np.unique(data[:, 0]):
#     ax.scatter(data[data[:, 0] == i, 1], data[data[:, 0] == i, 2], data[data[:, 0] == i, 3], s=3)
plt.title("clustered data with purity: %.3f" % purity(data[:, 0], label), fontsize=8)
for i in np.unique(label):
    ax.scatter(data[data[:, 0] == i, 1], data[data[:, 0] == i, 2], data[data[:, 0] == i, 3], s=3)
plt.show()
