# How to perform LDA
#Split training and test sets
data = iris #put the name of your dataset here

#Split the dataset in 70% training set and 30% test set
#You can choose other 30%
sample <- sample.int(n = nrow(data), size = floor(.70*nrow(data)), replace = F)
train <- data[sample, ]
test  <- data[-sample, ]

#Create your model
model = lda(Species ??? ., data=train) #instead of the "." put your explanatory variables: X1 + X2 + ... + Xn
# If you want to do qda just change lda( to qda(

prediction = predict(model, test)
predicted.values = prediction$class

#Compare the predicted values for the response in the test data 
#with the actual values for the response in the test data
table(prediction.class, test$response)  #Change response with the name of your response variable
mean(prediction.class == test$response)
