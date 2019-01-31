# evaluating LDA classifier
# install.packages("ISLR")
library(ISLR)
# install.packages("class") #If not installed yet
library (class)

train = (Smarket$Year < 2005)
Smarket.2005 = Smarket[!train, ]
Direction.2005 = Smarket$Direction[!train]

train.x = cbind(Smarket$Lag1, Smarket$Lag2)[train, ]
test.x = cbind(Smarket$Lag1, Smarket$Lag2)[!train, ]

# View(train.x)

train.Direction = Smarket$Direction[train]

knn.pred = knn(train.x, test.x, train.Direction, k = 10)
table(knn.pred, Direction.2005)
mean(knn.pred == Direction.2005)
