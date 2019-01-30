library(MASS)
View(cats)

# general linear model
glm.fit = glm(Sex ~ Bwt + Hwt, data = cats, family = binomial)
summary(glm.fit)

glm.probs = predict(glm.fit, type = "response")
glm.probs[1:10]

# linear classification
# install.packages("ggplot2")
library(ggplot2)
# help(iris) 
# View(iris)

# create the graph
plot1 <- ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species, shape = Species)) 
        + geom_point(size = 2)
# show
plot1

# create the graph
plot2 <- ggplot(iris, aes(x = Petal.Length, y = Petal.Width, color = Species, shape = Species)) 
      + geom_point(size = 2)

# show
plot2

# plot histogram
library(rcompanion)

# verify the assumption: Sepal.Width has a normal distribution
plotNormalHistogram(iris[which(iris$Species == "virginica"), ]$Sepal.Width)

plotNormalHistogram(iris[which(iris$Species == "virginica"), ]$Sepal.Length)

plotNormalHistogram(iris[which(iris$Species == "setosa"), ]$Petal.Width)

plotNormalHistogram(iris[which(iris$Species == "virginica"), ]$Petal.Length)

# using shapiro test: if p-value > 0.05 data is normally distributed
shapiro.test(iris[which(iris$Species == "setosa"), ]$Petal.Width)

# another test similiar to shapiro test, Levene's Test: assumption: data is homogenous
# install.packages("car")
library(car)
leveneTest(Sepal.Width ~ Species, data = iris, center = mean)


# evaluating LDA classifier
install.packages("ISLR")
library(ISLR)
train = (Smarket$Year < 2005)
Smarket.2005 = Smarket[!train, ]
# View(Smarket.2005)
Direction.2005 = Smarket$Direction[!train]

library(MASS)
# training
lda.fit = lda(Direction ~ Lag1 + Lag2, data = Smarket, subset = train)
lda.fit

# predict (model = lda.fit, test set = Smarket.2005)
# no need to normalize Lag1 and Lag2 (they have the same value types)
lda.predict = predict(lda.fit, Smarket.2005)

# see the result of classification
lda.class = lda.predict$class

# compare with real values
table(lda.class, Direction.2005)

# compute accuracy: how many items were predicted correctly / over all items
# (UP, Up) + (DOWN, DOWN) are correct predictions
mean(lda.class == Direction.2005)

################## LDA on IRIS dataset #################
# Linear    : lda() function
# when data points can be classified using a line

# quadratic : qda() function
# when data points are classified using a quadratic function

# split dataset into train and test
set.seed(100)
data = iris
sample <- sample.int(n = nrow(data), size = floor(0.7 * nrow(data)), replace = F)
train <- data[sample, ]
test <- data[-sample, ]
View(train)
View(test)

# model = lda(Species ~ ., data = train): Use "." as all variables, but they all have to be numeric
model = lda(
  Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, 
  data=train
)

# prediction
prediction = predict(model, test)

# show only the classification results
prediction$class

# show detailed prediction results
prediction

# evaluate
table(prediction$class, test$Species)
mean(prediction$class == test$Species)

