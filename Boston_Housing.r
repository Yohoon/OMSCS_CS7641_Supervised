suppressMessages ({
    library(dplyr)
    if (!require("tree")) install.packages("tree")
    library(tree)
    if (!require("caret")) install.packages("caret")
    library(caret)
    if (!require("gbm")) install.packages("gbm")
    library(gbm)
    if (!require("e1071")) install.packages("e1071")
    library(e1071)
    if (!require("class")) install.packages("class")
    library(class)
    if (!require("nnet")) install.packages("nnet")
    library(nnet)
    if (!require("ggplot2")) install.packages("ggplot2")
    library(ggplot2)
    if (!require("tidyr")) install.packages("tidyr")
    library(tidyr)
    if (!require("neuralnet")) install.packages("neuralnet")
    library(neuralnet)
    if (!require("adabag")) install.packages("adabag")
    library(adabag)
    library(caTools)
})

# import data
library(ISLR)
library(MASS)
df <- Boston
dim(Boston)

head(df)

sum(is.na(df))

summary(df)

for (i in 1:nrow(df)){

    if (df$medv[i] < 10){
        df$medv[i] <- 1
    } else if (df$medv[i] >= 10 & df$medv[i] < 20){
        df$medv[i] <- 2
    } else if (df$medv[i] >= 20 & df$medv[i] < 30){
        df$medv[i] <- 3
    } else if (df$medv[i] >= 30 & df$medv[i] < 40){
        df$medv[i] <- 4
    } else if (df$medv[i] >= 40 & df$medv[i] <= 50){
        df$medv[i] <- 5
    }
}
df$medv <- as.factor(df$medv)

summary(df)

# 5:1 train-test split

set.seed(2022) 

train = sample(1:nrow(df), nrow(df)*(4/5))
length(train)

#############################################################################################################
#                                  1. Decision tree with pruning                                            #
#############################################################################################################

# Whole tree with trainset before pruning

df2 <- df
tree.df=tree(medv~., df2, subset=train)

summary(tree.df)

plot(tree.df)
text(tree.df, pretty=0)

# See if pruning the tree will improve the results using Cross Validation

set.seed(5)

# use the cv.tree() function 
cv.df = cv.tree(tree.df)

# size: the number of terminal nodes, dev: deviance(SSE)
plot(cv.df$size ,cv.df$dev, type='b')   # type: l -> line, p -> point, b -> both

# The tree with 8 nodes is selected
# Pruning the tree

prune.df = prune.tree(tree.df, best = 8)
plot(prune.df)
text(prune.df , pretty = 0)

# test the tree
yhat1 = predict(prune.df, newdata=df2[-train,], type = 'class')
df.test = df2[-train ,"medv"]

t.tree <- confusionMatrix(yhat1, df.test)
t.tree$table
error <- mean(yhat1 != df.test)
print(paste('Accuracy =', 1-error))

#############################################################################################################
#                                      2. Neural network                                                    #
#############################################################################################################

# normalize non-factor columns
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
df3 <- df2
df3[,-ncol(df3)] <- normalize(df3[,-ncol(df3)])


# NN fitting
nn <- neuralnet(medv~., data=df3[train,], hidden=c(10,5), linear.output=T)

# test the model
pred <- predict(nn, df3[-train,])
yhat.nn <- as.factor(max.col(pred))

error <- mean(as.numeric(yhat.nn) != as.numeric(df3[-train,]$medv))
print(paste('Accuracy =', 1-error))

plot(nn, rep="best")

#############################################################################################################
#                                         3. Boosting                                                       #
#############################################################################################################

# boosting
set.seed(123)

adaboost <- boosting(medv~., data=df2[train,], boos=TRUE)

adaboost$importance

# test the model
yhat.boost = predict(adaboost, newdata = df2[-train ,], n.trees=500, type = 'response')
p.yhat.boost <- as.factor(apply(yhat.boost$prob, 1, which.max))

t.boost <- table(as.numeric(p.yhat.boost), as.numeric(df.test))
t.boost
error <- mean(as.numeric(p.yhat.boost) != as.numeric(df.test))
print(paste('Accuracy =', 1-error))

#############################################################################################################
#                                            4. SVM                                                         #
#############################################################################################################

# A cost argument allows us to specify the cost of a violation to the margin. 
# When the cost argument is small, then the margins will be wide 
# and many support vectors will be on the margin or will violate the margin
# The e1071 library includes a built-in function, tune(), 
# to perform cross- tune() validation to choose how much cost we will use.

set.seed(1)

# When kernel is linear
svm.tune.out = tune(svm, medv~., data = df2[train,], kernel ="linear", 
                    ranges = list(cost=c(0.001, 0.01, 0.1, 1,5,10,100)))

summary(svm.tune.out)   # best cost parameter is 1

# The tune() function stores the best model obtained, which can be accessed as follows:
bestmod = svm.tune.out$best.model
summary(bestmod)

# test the model

yhat.svm = predict(bestmod, df2[-train,])

t.svm <- confusionMatrix(yhat.svm, df.test)
t.svm$table
error <- mean(yhat.svm != df.test)
print(paste('Accuracy =', 1-error))

# When kernel is polynomial
set.seed(1)
svm.tune.out2 = tune(svm, medv~., data = df2[train,], kernel ="polynomial", 
                    ranges=list(cost=c(0.01,0.1,1,10,100,1000),
                            gamma=c(0.1,0.5,1,2,3,4) ))

summary(svm.tune.out2)   # the best cost is 10, and gamma is 0.1

bestmod2 = svm.tune.out$best.model
summary(bestmod2)

# test the model

yhat.svm2 = predict(bestmod2, df2[-train,])

t.svm2 <- confusionMatrix(yhat.svm2, df.test)
t.svm2$table
error <- mean(yhat.svm2 != df.test)
print(paste('Accuracy =', 1-error))

# When kernel is radial

set.seed(1)
svm.tune.out3 = tune(svm, medv~., data = df2[train,], kernel ="radial", 
                    ranges=list(cost=c(0.01,0.1,1,10,100,1000),
                            gamma=c(0.1,0.5,1,2,3,4)))

summary(svm.tune.out3)   # the best cost is 10, and gamma is 0.1

bestmod3 = svm.tune.out3$best.model
summary(bestmod3)

# test the model

yhat.svm3 = predict(bestmod3, df2[-train,])

t.svm3 <- confusionMatrix(yhat.svm3, df.test)
t.svm3$table
error <- mean(yhat.svm3 != df.test)
print(paste('Accuracy =', 1-error))

#############################################################################################################
#                                            5. KNN                                                         #
#############################################################################################################

set.seed(1)

# training error

errors <- NULL

for (k in 1:20){
    knn.df <- knn(train = df2[train,], test = df2[train,],
                      cl = df2[train,]$medv, k = k)
        
    errors[k] <- mean(knn.df != df2[train,]$medv)
}
train.knn.acc <- 1-errors

# test error

set.seed(1)

errors <- NULL

for (k in 1:20){
    
    cv.errors <- NULL
        
    knn.df <- knn(train = df2[train,], test = df2[-train,],
                      cl = df2[train,]$medv, k = k)
        
    error <- mean(knn.df != df2[-train,]$medv)
        
    # record error rate
    errors[k] <- error
}
    
test.knn.acc <- 1-errors

df.knn.acc <- as.data.frame(cbind(1:20, train.knn.acc, test.knn.acc))
head(df.knn.acc)

g <- ggplot(df.knn.acc, aes(x=V1)) + 
  geom_line(aes(y = train.knn.acc), color = "darkred") + 
  geom_line(aes(y = test.knn.acc), color="steelblue", linetype="twodash") 
g

which(test.knn.acc == max(test.knn.acc))


