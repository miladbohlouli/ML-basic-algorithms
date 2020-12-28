import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as ps
from pip._vendor.webencodings import labels
from scipy.cluster.hierarchy import dendrogram, linkage


def hierarchical_clustering_iterative(dissimilarity_matrix, method='single'):
    # In this step the indexes of the min is found, we will have to cmbine these two clusters and replace the
    #   the corresponding raw with a value according to the method we utilize

    initial_dis_matrix = np.copy(dissimilarity_matrix)
    num_data = dissimilarity_matrix.shape[0]
    Z = np.zeros((num_data - 1, 4))
    iteration = 0
    #   here we have an array to store the clusters, using this we will fill the Z matrix
    clusters = np.array(range(0, num_data))
    while True:
        min_value = np.min(dissimilarity_matrix[np.nonzero(dissimilarity_matrix)])
        min_index = np.where(dissimilarity_matrix == min_value)

        #   In this step we will have to update the z matrix
        #   in order to define the steps of converging the clusters
        Z[iteration, 0] = clusters[min_index[0][0]]
        Z[iteration, 1] = clusters[min_index[1][0]]
        Z[iteration, 2] = min_value

        #   In this step we will try to find the clusters and rearrange them
        value1 = clusters[min_index[0][0]]
        value2 = clusters[min_index[1][0]]
        clusters[clusters == value1] = num_data + iteration
        clusters[clusters == value2] = num_data + iteration
        Z[iteration, 3] = np.sum((clusters == num_data + iteration))
        iteration += 1

        if method == 'single':
            new_dists = np.min(dissimilarity_matrix[[min_index[0][0], min_index[1][0]], :], axis=0)

        elif method == 'complete':
            new_dists = np.max(dissimilarity_matrix[[min_index[0][0], min_index[1][0]], :], axis=0)


        elif method == 'average':
            new_dists = np.divide(np.sum(dissimilarity_matrix[[min_index[0][0], min_index[1][0]], :], axis=0), 2)

        else:
            print("!!!!!!The entered value for method is incorrect")

        new_dists[min_index[0][0]] = 0
        dissimilarity_matrix[min_index[0][0], :] = new_dists
        dissimilarity_matrix[:, min_index[0][0]] = np.transpose(new_dists)

        dissimilarity_matrix[min_index[1][0], :] = 0
        dissimilarity_matrix[:, min_index[1][0]] = 0

        if iteration == num_data - 1:
            return Z


#   Function to apply hierarchcal clustering on a data set
def hierarchical_clustering(input_data, method='single'):
    dissimilarity_matrix = np.zeros((input_data.shape[0], input_data.shape[0]))

    #   calculating the euclidean distance between the data for dissimilarity matrix
    for i in range(0, input_data.shape[0]):
        dissimilarity_matrix[i, :] = np.sqrt(np.sum(np.power(input_data[:, :] - input_data[i, :], 2), axis=1))

    return hierarchical_clustering_iterative(dissimilarity_matrix, method)


path = "datasets/question3"
data = np.array(ps.read_excel(os.path.join(path, "data_h.xlsx")))
plt.subplot(2, 2, 1)
plt.suptitle("Showing the final last 8 combination of hierarchical clustering", fontsize=10)
plt.title("Original data", fontsize=8)
plt.scatter(data[:, 0], data[:, 1], s=3)
plt.subplots_adjust(hspace=0.4)
z = hierarchical_clustering(data, 'single')
plt.subplot(2, 2, 2)
plt.title("single linkage", fontsize=8)
plt.yticks(fontsize=8)
dendrogram(z, p=8, truncate_mode='lastp', leaf_font_size=6)
z = hierarchical_clustering(data, 'complete')
plt.subplot(2, 2, 3)
plt.title("complete linkage", fontsize=8)
plt.yticks(fontsize=8)
dendrogram(z, p=8, truncate_mode='lastp', leaf_font_size=6)
z = hierarchical_clustering(data, 'average')
plt.subplot(2, 2, 4)
plt.title("average linkage", fontsize=8)
plt.yticks(fontsize=8)
dendrogram(z, p=8, truncate_mode='lastp', leaf_font_size=6)

plt.show()
