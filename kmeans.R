library(MASS)
set.seed(1)

# use dataset iris
data = iris

# number of clusters - 5, repeat for 20 times, using euclidian distance
km.out = kmeans(data[1:4], 5, nstart = 20)
km.cluster = km.out$cluster

# see the observations in clusters
# (between_SS / total_SS) shows the distance of clusters
km.out
