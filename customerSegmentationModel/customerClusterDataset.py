import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import KMeans

#Price of the Olay products
x = [180,275,339,365,369,379,397,399,499,645,818,849,999,1267]
#Array to hold the number of people buying it
y=[]

#Numpy array to hold both x and y coordinates
nX = []

#Fill in the above arrays
for i in range(0,14):
    y.append(round((math.exp(-(x[i])/100) * 1000 + 100),2))
    nX.append([x[i],y[i]])

print(x)
#print(nX)

#Find the K means of cluster 3 on array
n_clusters = 3
kmeans = KMeans(n_clusters=3)
kmeans.fit(nX)

#Find the cluster and the label each value in list
centroids = kmeans.cluster_centers_
labels = kmeans.labels_
print("centroids : ",centroids)
print("labels : ",labels)

#color coding to plot the values
colors = ["g.","r.","c.","y.","g.","r.","c.","y.","g.","r.","c.","y.""g.","r."]

#Plot the points on the graph
for i in range(len(nX)):
    #print("coordinate:",nX[i], "label:", labels[i])
    plt.plot(nX[i][0], nX[i][1], colors[labels[i]], markersize = 10)

# squared distance to cluster center
X_dist = kmeans.transform(nX)
#print("X_dist ",X_dist)

#Logic to give the customer next higher value product

#1.Find the furhtest points distance from cluster_centers
furhtest_dist = []
for i in range(1, len(x)):
    if labels[i-1] != labels[i]:
        furhtest_dist.append(nX[i-1][0] - centroids[labels[i-1]][0])
        #print("nX",nX[i-1][0]," - centriods",centroids[labels[i-1]][0],"further_dist ", furhtest_dist)
    elif i == len(x) - 1:
        furhtest_dist.append(nX[i][0] - centroids[labels[i]][0])
        #print("nX",nX[i][0]," - centriods",centroids[labels[i]][0],"further_dist ", furhtest_dist)


updatedProduct = []

def closest(lst, K):

     lst = np.asarray(lst)
     idx = (np.abs(lst - K)).argmin()
     if lst[idx] > K:
         return lst[idx]
     elif lst[idx] < K and idx != len(lst) -1:
         return lst[idx + 1]
     else:
         return lst[idx]




for j in range(len(x)):
    val = abs(centroids[labels[j]][0] - x[j])
    #farDistance = y[min(range(len(y)), key = lambda i: abs(y[i]-(x[j] + val/2))]
    if val >= furhtest_dist[labels[j]]:
        print("j : ",j,"val :",x[j] + val/2,"y_min : ",closest(x, x[j] + val/2))
        updatedProduct.append(closest(x, x[j] + val/2))
    else:
        print("j : ",j,"val :",x[j] + furhtest_dist[labels[j]]/2,"y_min : ",closest(x,  x[j] + furhtest_dist[labels[j]]/2))
        updatedProduct.append(closest(x, x[j] + furhtest_dist[labels[j]]/2))

print(updatedProduct)



plt.scatter(centroids[:, 0],centroids[:, 1], marker = "x", s=150, linewidths = 5, zorder = 10)
plt.show()
