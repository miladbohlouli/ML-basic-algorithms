from matplotlib import pyplot as plt
import numpy as np
import sklearn.cluster as clus

c=5
data = np.genfromtxt("datasets/question1/data_kmeans_5.txt", delimiter=',')

kmeans = clus.KMeans(n_clusters=c).fit(data)
labels = kmeans.labels_
centers = kmeans.cluster_centers_
plt.title("clustered data set of data_kmeans_5 using library with k = " + str(c) , fontsize=11)
for i in range(0, c):
    plt.scatter(data[labels == i, 0], data[labels == i, 1], s=2)
    plt.scatter(centers[i, 0], centers[i, 1], color='black', marker="X")
plt.savefig("../Results/library_" + str(c))



