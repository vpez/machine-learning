install.packages("class") #If not installed yet
library(class)

View(mtcars)
data = mtcars

library(ggplot2)

plot_cars = ggplot(data,aes(x=cyl,y=hp)) + geom_point()
plot_cars

plot1 <- ggplot(mtcars, 
                aes(x = cyl, y = hp, color = vs, shape = vs)) + geom_point(size = 2)
plot1

sample <- sample.int(n = nrow(data), size = floor(0.7 * nrow(data)), replace = F)

train = data[sample, ]
test = data[-sample, ]
#View(train)
#View(test)
engine.vs = test$vs
#View(engine.vs)

train.X=cbind(data$cyl, data$hp)[sample, ]
test.X=cbind (data$cyl, data$hp)[-sample, ]
train.engine.vs = train$vs

knn.pred = knn(train.X, test.X, train.engine.vs, k=4)
table(knn.pred, engine.vs)
mean(knn.pred == engine.vs)

