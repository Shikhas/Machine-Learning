import pandas
import numpy
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import math
import matplotlib.pyplot as plt

Cars_data = pandas.read_csv("cars.csv")

nCars = Cars_data.shape[0]
#print(nCars)

print(Cars_data.columns) # to investigate all the column names in given data

trainData = numpy.reshape(numpy.asarray(Cars_data[['Horsepower','Weight']]), (nCars, 2))
print(trainData)
KClusters =15
kmeans = cluster.KMeans(n_clusters=KClusters, random_state=60616).fit(trainData)


silhouette_avg = metrics.silhouette_score(trainData, kmeans.labels_)
print(silhouette_avg)

TotalWCSS  = numpy.zeros(KClusters)
nClusters =numpy.zeros(KClusters)
Silhouette =numpy.zeros(KClusters)
Elbow =numpy.zeros(KClusters)

#import ipdb; ipdb.set_trace()
#Implementing Elbow method and Silhouette method to find optimal number of clusters

for c in range(KClusters):
    KCluster = c+1
    nClusters[c] = KCluster

    kmeans = cluster.KMeans(n_clusters=KCluster, random_state=60616).fit(trainData)

    if (1 < KCluster & KCluster < 16):
        Silhouette[c] = metrics.silhouette_score(trainData, kmeans.labels_)
    else:
        Silhouette[c] = numpy.NaN

    WCSS = numpy.zeros(KCluster)
    nC = numpy.zeros(KCluster)

    for i in range(nCars):
        k = kmeans.labels_[i]
        nC[k] += 1
        #diff = Cars_data[i] - kmeans.cluster_centers_[k]
        diff = math.sqrt(math.pow((trainData[i][0] - kmeans.cluster_centers_[k][0]), 2) +
                         math.pow((trainData[i][1] - kmeans.cluster_centers_[k][1]), 2))
        WCSS[k] += diff*diff

    Elbow[c] = 0
    for k in range(KCluster):
        Elbow[c] += WCSS[k] / nC[k]
        TotalWCSS[c] += WCSS[k]

    print("Cluster Assignment:", kmeans.labels_)
    for k in range(KCluster):
        print("Cluster ", k)
        print("Centroid = ", kmeans.cluster_centers_[k])
        print("Size = ", nC[k])
        print("Within Sum of Squares = ", WCSS[k])
        print(" ")

# Listing the Elbow and Silhouette values for 1-cluster to 15-cluster solutions
print("The Elbow and Silhouette values for 1-cluster to 15-cluster solutions are as follows:")
print("N Clusters\t Total WCSS\t Elbow Value\t Silhouette Value:")
for c in range(15):
    print('{:.0f} \t {:.4f} \t {:.4f} \t {:.4f}'
          .format(nClusters[c], TotalWCSS[c], Elbow[c], Silhouette[c]))

# Plotting the Elbow  values against number of clusters
plt.plot(nClusters, Elbow, linewidth=2, marker='o')
plt.grid(True)
plt.title("Elbow values versus number of clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Elbow Value")
plt.xticks(numpy.arange(1, 16, step=1))
plt.show()

# Plotting the Silhouette values against number of clusters
plt.plot(nClusters, Silhouette, linewidth=2, marker='o')
plt.grid(True)
plt.title("Silhouette values versus number of clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Value")
plt.xticks(numpy.arange(1, 16, step=1))
plt.show()


# After observing graph I think K = 4



