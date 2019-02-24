import pandas
import math
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.neighbors as neigh
import numpy
from numpy import linalg as LA

Spiral_data = pandas.read_csv("Spiral.csv")
nObs = Spiral_data.shape[0]
print(Spiral_data.columns)

# Generating Scatter plot for visual inspection
plt.scatter(Spiral_data["x"],Spiral_data["y"])
plt.title("Scatter plot of X versus Y")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()

# The k from visualization can be 2

# Applying the K-mean algorithm directly using 2 number of clusters.
trainData = Spiral_data[['x','y']]
kmeans = cluster.KMeans(n_clusters=2, random_state=60616).fit(trainData)
print("Cluster Centroids = \n", kmeans.cluster_centers_)
Spiral_data['KMeanCluster'] = kmeans.labels_
for i in range(2):
    print("Cluster Label = ", i)
    print(Spiral_data.loc[Spiral_data['KMeanCluster'] == i])
#Regenerateing the scatterplot using the K-mean cluster identifier to control the color scheme
plt.figure()
plt.scatter(Spiral_data['x'], Spiral_data['y'], c = Spiral_data['KMeanCluster'])
plt.title("Scatter plot after K-Means Clustering")
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

# Three nearest neighbors as obtained by visual inspectipn
kNNSpec = neigh.NearestNeighbors(n_neighbors=3, algorithm='brute', metric='euclidean')
nbrs = kNNSpec.fit(trainData)
d3, i3 = nbrs.kneighbors(trainData)

# Retrieve the distances among the observations
distObject = neigh.DistanceMetric.get_metric('euclidean')
distances = distObject.pairwise(trainData)

# Create the Adjacency and the Degree matrices
Adjacency = numpy.zeros((nObs, nObs))
Degree = numpy.zeros((nObs, nObs))

for i in range(nObs):
    for j in i3[i]:
        if (i <= j):
            Adjacency[i, j] = math.exp(- distances[i][j])
            Adjacency[j, i] = Adjacency[i, j]

for i in range(nObs):
    sum = 0
    for j in range(nObs):
        sum += Adjacency[i, j]
    Degree[i, i] = sum

#Laplacian Matrix defined
Lmatrix = Degree - Adjacency

evals, evecs = LA.eigh(Lmatrix)

# Sequence plot of the first 9 eigenvalues to determine the number of clusters
plt.scatter(numpy.arange(0, 9, 1), evals[0:9, ])
plt.title("Sequence Plot of first 9 eigenvalues with 3 nearest neighbours.")
plt.xlabel('Sequence')
plt.ylabel('Eigenvalue')
plt.show()

######## In plot obtained, the jump appears to be from 4 to 5 ###########

# Inspecting the values of the first two eigenvectors
Z = evecs[:, [0, 1]]

plt.scatter(Z[[0]], Z[[1]])
plt.title("Plotting first two eigenvectors.")
plt.xlabel('Z[0]')
plt.ylabel('Z[1]')
plt.show()

# Applying the K-mean algorithm on first two eigenvectors that correspond to the first two smallest eigenvalues.
kmeans_spectral = cluster.KMeans(n_clusters=2, random_state=60616).fit(Z)
Spiral_data['SpectralCluster'] = kmeans_spectral.labels_

#Regenerating the scatterplot using the K-mean cluster identifier to control the color scheme
plt.scatter(Spiral_data['x'], Spiral_data['y'], c=Spiral_data['SpectralCluster'])
plt.title("Scatter plot after Spectral Clustering")
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

# Conclusion: as data is not compact but its connected, that’s why k-means algorithm does not works well while
# Spectral clustering gives us desired results. Spectral clustering needs to be implemented for data with connectivity
#  as it clusters the data points based on their connectivity while K-means is clusters based on compactness of data
