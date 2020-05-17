import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import KMeans


x = [180,275,339,365,369,379,397,399,499,645,818,849,999,1267]
y=[]

nX = []
#y = [455,423,300,290,286,276,265,264,164,62,42,40,32,16]

for i in range(0,14):
    y.append(round((math.exp(-(x[i])/100) * 1000 + 100),2))
    nX.append([x[i],y[i]])

#print(nX)

kmeans = KMeans(n_clusters=2)
kmeans.fit(nX)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print(centroids)
print(labels)

colors = ["g.","r.","c.","y.","g.","r.","c.","y.","g.","r.","c.","y.""g.","r."]

for i in range(len(nX)):
    print("coordinate:",nX[i], "label:", labels[i])
    plt.plot(nX[i][0], nX[i][1], colors[labels[i]], markersize = 10)


plt.scatter(centroids[:, 0],centroids[:, 1], marker = "x", s=150, linewidths = 5, zorder = 10)

plt.show()
