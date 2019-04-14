from scipy.stats import iqr
import pandas
import numpy
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import math
import matplotlib.pyplot as plt

pothole_data = pandas.read_csv("Data/ChicagoCompletedPotHole.csv")

n_pothole = pothole_data.shape[0]
print(n_pothole)

print(pothole_data.columns)  # to investigate all the column names in given data
#import ipdb;ipdb.set_trace()
tranfd_pothole_data_col1 = numpy.log(pothole_data['N_POTHOLES_FILLED_ON_BLOCK'])
#print(tranfd_pothole_data_col1)

tranfd_pothole_data_col2= numpy.log(1 + pothole_data['N_DAYS_FOR_COMPLETION'])

#print(tranfd_pothole_data_col2)
transfd_data = pandas.DataFrame(pothole_data,columns=["LATITUDE","LONGITUDE"])
transfd_data['log_pothole_filled'] = tranfd_pothole_data_col1

transfd_data['log_days'] = tranfd_pothole_data_col2

#print(transfd_data)

trainData = numpy.reshape(numpy.asarray(transfd_data),(n_pothole,4))
KClusters = 10
kmeans = cluster.KMeans(n_clusters=KClusters, random_state=20190327).fit(trainData)

silhouette_avg = metrics.silhouette_score(trainData, kmeans.labels_)
print(silhouette_avg)

TotalWCSS = numpy.zeros(KClusters)
nClusters = numpy.zeros(KClusters)
Silhouette = numpy.zeros(KClusters)
Elbow = numpy.zeros(KClusters)

# import ipdb; ipdb.set_trace()
# Implementing Elbow method and Silhouette method to find optimal number of clusters

for c in range(1,KClusters):
    KCluster = c + 1
    nClusters[c] = KCluster

    kmeans = cluster.KMeans(n_clusters=KCluster, random_state=20190327).fit(trainData)

    if (1 < KCluster & KCluster < 11):
        Silhouette[c] = metrics.silhouette_score(trainData, kmeans.labels_)
    else:
        Silhouette[c] = numpy.NaN

    WCSS = numpy.zeros(KCluster)
    nC = numpy.zeros(KCluster)

    for i in range(n_pothole):
        k = kmeans.labels_[i]
        nC[k] += 1
        #import ipdb; ipdb.set_trace()
        diff = math.sqrt(math.pow((trainData[i][0] - kmeans.cluster_centers_[k][0]), 2) +
                         math.pow((trainData[i][1] - kmeans.cluster_centers_[k][1]), 2)+
                         math.pow((trainData[i][2] - kmeans.cluster_centers_[k][2]), 2)+
                         math.pow((trainData[i][3] - kmeans.cluster_centers_[k][3]), 2))
        WCSS[k] += diff * diff

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

# Listing the Elbow and Silhouette values for 2-cluster to 10-cluster solutions
print("The Elbow and Silhouette values for 2-cluster to 10-cluster solutions are as follows:")
print("N Clusters\t Total WCSS\t Elbow Value\t Silhouette Value:")
for c in range(1,10):
    print('{:.0f} \t {:.4f} \t {:.4f} \t {:.4f}'
          .format(nClusters[c], TotalWCSS[c], Elbow[c], Silhouette[c]))

# Plotting the Elbow  values against number of clusters
plt.plot(nClusters[1:], Elbow[1:], linewidth=2, marker='o')
plt.grid(True)
plt.title("Elbow values versus number of clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Elbow Value")
plt.xticks(numpy.arange(2, 11, step=1))
plt.show()

# Plotting the Silhouette values against number of clusters
plt.plot(nClusters[1:], Silhouette[1:], linewidth=2, marker='o')
plt.grid(True)
plt.title("Silhouette values versus number of clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Value")
plt.xticks(numpy.arange(2, 11, step=1))
plt.show()


plt.plot(nClusters[1:], TotalWCSS[1:], linewidth=2, marker='o')
plt.grid(True)
plt.title("TWCSS values versus number of clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("TWCSS Value")
plt.xticks(numpy.arange(2, 11, step=1))
plt.show()


# Load the TREE library from SKLEARN
from sklearn import tree
classTree = tree.DecisionTreeClassifier(criterion='gini', max_depth=2, random_state=20190327)
trainData_2= pandas.DataFrame(pothole_data,columns=["N_POTHOLES_FILLED_ON_BLOCK","N_DAYS_FOR_COMPLETION","LATITUDE","LONGITUDE"])

kmeans = cluster.KMeans(n_clusters=4, random_state=20190327).fit(trainData)
pothole_DT = classTree.fit(trainData_2, kmeans.labels_)

color_map = ["yellow", "blue", "green", "red"]
axs = plt.gca()
for label in range(4):
    members = pothole_data[kmeans.labels_ == label]
    axs.scatter(members['LONGITUDE'], members['LATITUDE'], c=color_map[label], label=f"Cluster-{label}", s=0.5)
axs.legend()
#axs.set_aspect(aspect=1)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()


print('Accuracy of Decision Tree classifier on training set: {:.6f}' .format(classTree.score(trainData_2, kmeans.labels_)))

import graphviz
dot_data = tree.export_graphviz(pothole_DT,
                                out_file=None,
                                impurity = True, filled = True,
                                feature_names = ['N_POTHOLES_FILLED_ON_BLOCK', 'N_DAYS_FOR_COMPLETION', 'LATITUDE','LONGITUDE'],
                                class_names = ['Cluster 0', 'Cluster 1','Cluster 2','Cluster 3'])

graph = graphviz.Source(dot_data)

#print(graph)
graph.render('mid-test', ".")

result_df = pandas.DataFrame()
result_df["true labels"] = kmeans.labels_
result_df["predicted labels"] = classTree.predict(trainData_2)
sqrd_sum = 0
for i in range(n_pothole):
    sqrd_sum = sqrd_sum + math.pow((result_df["true labels"][i] - result_df["predicted labels"][i]),2)

print(sqrd_sum)
print("The Root Average Squared Error is ", math.sqrt(sqrd_sum/n_pothole))

probs = classTree.predict_proba(trainData_2)
dd = pandas.DataFrame()

dd['0'] = probs[:, 0]
dd['1'] = probs[:, 1]
dd['2'] = probs[:, 2]
dd['3'] = probs[:, 3]

ase = 0
for i in range(result_df.shape[0]):
    for categ in range(4):
        is_correct_categ = 1 if result_df["predicted labels"][i] == categ else 0
        ase += (is_correct_categ - dd[f'{categ}'][i]) ** 2

print(ase)



data_points_iqr = numpy.array([0.1811, 0.0775, 0.1279, 0.0045, 0.0001, 0.9457, 0.0021, 0, 0.0005, 0.7305, 0.8936])
#for i in data_points_iqr:
print("The IQR for the series of given numbers is {} " .format(iqr(data_points_iqr)))