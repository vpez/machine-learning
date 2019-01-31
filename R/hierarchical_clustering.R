library(MASS)
set.seed(1)

# use dataset iris
data = iris

hc.out = hclust(dist(data[1:4]))
plot(hc.out, labels = as.numeric(iris$Species))

# cut the dendrogram at level 10
hc.clusters = cutree(hc.out, 10)
hc.clusters
