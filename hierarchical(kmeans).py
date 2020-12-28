import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os

def kmeans(input_data, k):
    rand = np.random.randint(low = 0, high = input_data.shape[0], size=k)
    new_centers = input_data[rand]
    distance = np.zeros((input_data.shape[0], k))
    iteration = 0
    while True:
        error = 0
        iteration += 1
        old_centers = new_centers.copy()
        for z in range(0, k):
            distance[:, z] = np.sqrt(np.sum(np.power(np.subtract(input_data, old_centers[z]), 2), axis=1))
        clusters = np.argmin(distance, axis=1)
        for z in range(0, k):
            new_centers[z] = np.sum(input_data[clusters == z], axis=0) / np.sum(clusters == z)

        for z in range(k):
            error += np.sum(np.sqrt(np.sum(np.power(np.subtract(input_data[clusters == z], new_centers[z]), 2), axis=1)),
                            axis=0)
        if np.array_equal(new_centers, old_centers):
            break
    return clusters


def top_down(complete_data):
    num_data = complete_data.shape[0]
    clusters = np.zeros(num_data)
    chosen_cluster = 0
    c = 0
    plt.subplot(3, 4, 1), plt.title("Original data", fontsize=8)
    plt.suptitle("Result of the top down hierarchical clustering using kmeans", fontsize=10)
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    plt.scatter(data[:, 0], data[:, 1], s=2)
    while c <= 8:
        binary_clusters = kmeans(complete_data[clusters == chosen_cluster], 2)
        val, counts = np.unique(binary_clusters, return_counts=True)
        max_num = np.max(clusters)
        clusters[clusters == chosen_cluster] = binary_clusters + max_num + 1
        chosen_cluster = val[np.argmax(counts)] + max_num + 1
        print("clusters: " + str(np.unique(clusters)))
        print("chosen cluster: " + str(chosen_cluster))
        plt.subplot(3, 4, c+2), plt.title("clustered data into %d clusters" %(c+2), fontsize=8)
        for i in np.unique(clusters):
            plt.scatter(complete_data[clusters == i, 0], complete_data[clusters == i, 1], s=3)
        c += 1
    plt.show()
    return


path = "datasets/question3"
data = np.array(pd.read_excel(os.path.join(path, "data_h.xlsx")))
top_down(data)
