import os
import numpy as np
from matplotlib import pyplot as plt

def kmeans(input_data, k):
    rand = np.random.randint(low = 0, high = input_data.shape[0], size=k)
    new_centers = input_data[rand]
    distance = np.zeros((input_data.shape[0], k))
    iteration = 0
    db_matrix = np.zeros((k, k))
    SSE = []
    DB = []

    while True:
        error = 0
        iteration += 1
        old_centers = new_centers.copy()
        for z in range(0, k):
            distance[:, z] = np.sqrt(np.sum(np.power(np.subtract(input_data, old_centers[z]), 2), axis=1))
        clusters = np.argmin(distance, axis=1)
        for z in range(0,k):
            new_centers[z] = np.sum(input_data[clusters == z], axis=0) / np.count_nonzero(clusters == z)

        for z in range(k):
            error += np.sum(np.sqrt(np.sum(np.power(np.subtract(input_data[clusters == z], new_centers[z]), 2), axis=1)),
                            axis=0)
        for i in range(k):
            for j in range(i+1, k):
                norm_i = np.sum(clusters == i)
                norm_j = np.sum(clusters == j)
                db_matrix[i, j] = (np.sum(np.sqrt(np.sum(np.power(np.subtract(input_data[clusters == i], new_centers[i]), 2), axis=1)), axis=0) / norm_i + \
                                   np.sum(np.sqrt(np.sum(np.power(np.subtract(input_data[clusters == j], new_centers[j]), 2), axis=1)), axis=0) / norm_j) / \
                                  np.sqrt(np.sum(np.power(np.subtract(new_centers[i], new_centers[j]), 2)))

        db = np.sum(np.max(db_matrix, axis=1), axis=0) / k
        SSE.append(error)
        DB.append(db)

        for z in range(k):
            plt.suptitle("Different clusters and centers in each iteration")
            plt.subplot(10, 10, iteration), plt.title("iteartion: " + str(iteration), fontsize=8)
            plt.subplots_adjust(hspace=0.5, wspace=0.3), plt.xticks([]), plt.yticks([])
            plt.scatter(data[clusters == z, 0], data[clusters == z, 1], s=3)
            plt.scatter(new_centers[z, 0], new_centers[z, 1], marker="X", color = 'black', s=2)


        if np.array_equal(new_centers, old_centers):
            plt.savefig("../Results/kmean_5_30(1)")
            break
    return clusters, SSE, DB

c = 30
path = "datasets/question1"
file_name = "data_kmeans_5.txt"
data = np.genfromtxt(os.path.join(path, file_name), delimiter=',')
clusters, SSE, DB = kmeans(data, c)
plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.suptitle("Results for the data set of " + file_name, fontsize=12)
plt.subplot(2, 2, 1)
plt.scatter(data[:, 0], data[:, 1], s=3, color='red'), plt.title("Original data", fontsize=10)
plt.subplot(2, 2, 2)
plt.plot(range(0, SSE.__len__()), SSE), plt.xticks(range(0, SSE.__len__()), fontsize=6)
plt.title("SSE in each iteration", fontsize=10)
plt.subplot(2, 2, 3)
plt.plot(range(0, DB.__len__()), DB), plt.xticks(range(0, DB.__len__()), fontsize=6)
plt.title("DB in each iteration", fontsize=10)
plt.subplot(2, 2, 4)

for i in range(c):
    plt.scatter(data[clusters == i, 0], data[clusters == i, 1], s=3)
    plt.scatter(np.sum(data[clusters == i], axis=0)[0] / np.sum(clusters == i),
                np.sum(data[clusters == i], axis=0)[1] / np.sum(clusters == i),
                marker='X', color='black',s=4)
plt.show()
# plt.savefig("../Results/kmean_5_30(2)")
