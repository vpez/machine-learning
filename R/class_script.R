library(MASS)
data(cars)
View(cars)
summary(cars)

model = lm(dist ~ speed, data=cars)

# dist = intercept + coeff * speed
# intercept = -17.5791, coeff = 3.9324
# R-squared = 0.6511 -> can explain 65% of variability
summary(model)

# add the predicted value in a new column ($predicted)
cars$predicted = predict(model)
View(cars)

# plot the data points
plot(cars$speed, cars$dist)

# correlation
cor(cars$speed, cars$dist)

# residuals of the model
cars$residuals = residuals(model)
cars$studres = studres(model)
View(cars)

plot(cars$predicted, cars$studres)
abline(h=2)
abline(h=-2)

library(rcompanion)
plotNormalHistogram(cars$studres)

# test the p-value:
# What the fuck is p-value:
# https://www.dummies.com/education/math/statistics/what-a-p-value-tells-you-about-statistical-data/
shapiro.test(cars$residuals)

# you like when residuals are ZERO
plot(cars$predicted, cars$studres)
abline(h=0)

library(car)
durbinWatsonTest(model)

# example 2: with multiple predictor variables
cars = Cars93
View(cars)

model = lm(Price ~ Horsepower + Width + Wheelbase, data = cars)
summary(model)

# the predictor variables with probability less than 0.005 are significant. => They have starts
# how to choose multiple variables?

# 1. forward selection
# start with the null model (no variables)
# try by adding variables and see the summary of the model
#   - add siginificant variables
#   - ignore non-significant variables

# 2. backward selection: add everything, remove the ones that are non-significant
cars$predicted = predict(model)

model = lm(Price ~ Horsepower + Width, data = cars)
summary(model)

model = lm(Price ~ Horsepower + EngineSize, data = cars)
summary(model)

model = lm(Price ~ Horsepower + Length, data = cars)
summary(model)

model = lm(Price ~ Horsepower + Wheelbase, data = cars)
summary(model)

# qualitative predictors: Manufacturers, Type etc.
# assign dummy variables to each qualitative value (category). e.g. 1 = male, 2 = female

# non-linear: poly(variable, power): poly(Hoursepower, 2)
model = lm(Price ~ poly(Horsepower, 2) + Wheelbase, data = cars)
summary(model)

# install "datasets" and see more examples

# to import your own dataset: myData = read.csv("path")
