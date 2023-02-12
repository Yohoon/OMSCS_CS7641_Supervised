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

# For Migraine/Headache Classification
# from https://www.kaggle.com/datasets/weinoose/migraine-classification

# Attribute Information:
#1) Age: Patient's age
#2) Duration: duration of symptoms in last episode in days
#3) Frequency: Frequency of episodes per month
#4) Location: Unilateral or bilateral pain location (None - 0, Unilateral - 1, Bilateral - 2)
#5) Character: Throbbing or constant pain (None - 0, Thobbing - 1, Constant - 2)
#6) Intensity: Pain intensity, i.e., mild, medium, or severe (None - 0, Mild - 1, Medium - 2, Severe - 3)
#7) Nausea: Nauseous feeling (Not - 0, Yes - 1)
#8) Vomit: Vomiting (Not - 0, Yes - 1)
#9) Phonophobia: Noise sensitivity (Not - 0, Yes - 1)
#10) Photophobia: Light sensitivity (Not - 0, Yes - 1)
#11) Visual: Number of reversible visual symptoms
#12) Sensory: Number of reversible sensory symptoms
#13) Dysphasia: Lack of speech coordination (Not - 0, Yes - 1)
#14) Dysarthria: Disarticulated sounds and words (Not - 0, Yes - 1)
#15) Vertigo: Dizziness (Not - 0, Yes - 1)
#16) Tinnitus: Ringing in the ears (Not - 0, Yes - 1)
#17) Hypoacusis: Hearing loss (Not - 0, Yes - 1)
#18) Diplopia: Double vision (Not - 0, Yes - 1)
#19) Visual defect: Simultaneous frontal eye field and nasal field defect and in both eyes (Not - 0, Yes - 1)
#20) Ataxia: Lack of muscle control (Not - 0, Yes - 1)
#21) Conscience: Jeopardized conscience (Not - 0, Yes - 1)
#22) Paresthesia: Simultaneous bilateral paresthesia (Not - 0, Yes - 1)
#23) DPF: Family background (Not - 0, Yes - 1)
#24) Type: Diagnosis of migraine type 


# import data
df <- read.csv("data.csv")
dim(df)

# factorize some columns
df <- df %>% mutate_at(vars(4:10,13:24), as.factor)  
df$Type_num <- as.factor(as.numeric(df$Type))  # add a column with factorized numeric Type 

df2 <- subset(df, select=-c(Type))  # remove non-numeric 'Type' column
head(df2)
str(df2)



summary(df2)   # Ataxia has only one value, so let's remove.

df2 <- subset(df2, select=-c(Ataxia))

# histograms and correlation graphs only for numericals

ggplot(gather(df2[,c(1:3, 11:12)]), aes(value)) + 
    geom_histogram(bins = 10) + 
    facet_wrap(~key, scales = 'free_x')

# 3:1 train-test split

set.seed(2023) 

train = sample(1:nrow(df2), nrow(df2)*(3/4))
length(train)

#############################################################################################################
#                                  1. Decision tree with pruning                                            #
#############################################################################################################

# Whole tree with trainset before pruning

tree.df=tree(Type_num~., df2, subset=train)
tree.df

summary(tree.df)

plot(tree.df)
text(tree.df, pretty=0)

# See if pruning the tree will improve the results using Cross Validation

# use the cv.tree() function 
cv.df = cv.tree(tree.df)

# size: the number of terminal nodes, dev: deviance(SSE)
plot(cv.df$size ,cv.df$dev, type='b')   # type: l -> line, p -> point, b -> both

# The tree with 9 nodes is selected
# Pruning the tree

prune.df = prune.tree(tree.df, best = 9)
plot(prune.df)
text(prune.df , pretty = 0)

# compare test result between unpruned tree and pruned tree

# with unpruned tree
yhat1 = predict(tree.df, newdata=df2[-train,], type = 'class')
df.test = df2[-train ,"Type_num"]

t.tree <- confusionMatrix(yhat1, df.test)
t.tree$table
error <- mean(yhat1 != df.test)
print(paste('Accuracy =', 1-error))


# with pruned tree
yhat2 = predict(prune.df, newdata=df2[-train,], type = 'class')

t.tree.pruned <- confusionMatrix(yhat2, df.test)
t.tree.pruned$table
error <- mean(yhat2 != df.test)
print(paste('Accuracy =', 1-error))

#############################################################################################################
#                                      2. Neural network                                                    #
#############################################################################################################

df3 <- df2
df3$Location_None <- ifelse(df3$Location == '0', 1, 0)
df3$Location_Unilateral <- ifelse(df3$Location == '1', 1, 0)
df3$Location_Bilateral <- ifelse(df3$Location == '2', 1, 0)
df3$Character_none <- ifelse(df3$Character == '0', 1, 0)
df3$Character_Thobbing <- ifelse(df3$Character == '1', 1, 0)
df3$Character_Constant <- ifelse(df3$Character == '2', 1, 0)
df3$Intensity_None <- ifelse(df3$Intensity == '2', 1, 0)
df3$Intensity_Mild <- ifelse(df3$Intensity == '0', 1, 0)
df3$Intensity_Medium <- ifelse(df3$Intensity == '1', 1, 0)
df3$Intensity_Severe <- ifelse(df3$Intensity == '2', 1, 0)
df3$Nausea_Not <- ifelse(df3$Nausea == '0', 1, 0)
df3$Nausea_Yes <- ifelse(df3$Nausea == '1', 1, 0)
df3$Vomit_Not <- ifelse(df3$Vomit == '0', 1, 0)
df3$Vomit_Yes <- ifelse(df3$Vomit == '1', 1, 0)
df3$Phonophobia_Not <- ifelse(df3$Phonophobia == '0', 1, 0)
df3$Phonophobia_Yes <- ifelse(df3$Phonophobia == '1', 1, 0)
df3$Photophobia_Not <- ifelse(df3$Photophobia == '0', 1, 0)
df3$Photophobia_Yes <- ifelse(df3$Photophobia == '1', 1, 0)
df3$Dysphasia_Not <- ifelse(df3$Dysphasia == '0', 1, 0)
df3$Dysphasia_Yes <- ifelse(df3$Dysphasia == '1', 1, 0)
df3$Dysarthria_Not <- ifelse(df3$Dysarthria == '0', 1, 0)
df3$Dysarthria_Yes <- ifelse(df3$Dysarthria == '1', 1, 0)
df3$Vertigo_Not <- ifelse(df3$Vertigo == '0', 1, 0)
df3$Vertigo_Yes <- ifelse(df3$Vertigo == '1', 1, 0)
df3$Tinnitus_Not <- ifelse(df3$Tinnitus == '0', 1, 0)
df3$Tinnitus_Yes <- ifelse(df3$Tinnitus == '1', 1, 0)
df3$Hypoacusis_Not <- ifelse(df3$Hypoacusis == '0', 1, 0)
df3$Hypoacusis_Yes <- ifelse(df3$Hypoacusis == '1', 1, 0)
df3$Diplopia_Not <- ifelse(df3$Diplopia == '0', 1, 0)
df3$Diplopia_Yes <- ifelse(df3$Diplopia == '1', 1, 0)
df3$Defect_Not <- ifelse(df3$Defect == '0', 1, 0)
df3$Defect_Yes <- ifelse(df3$Defect == '1', 1, 0)
df3$Conscience_Not <- ifelse(df3$Conscience == '0', 1, 0)
df3$Conscience_Yes <- ifelse(df3$Conscience == '1', 1, 0)
df3$Paresthesia_Not <- ifelse(df3$Paresthesia == '0', 1, 0)
df3$Paresthesia_Yes <- ifelse(df3$Paresthesia == '1', 1, 0)
df3$DPF_Not <- ifelse(df3$DPF == '0', 1, 0)
df3$DPF_Yes <- ifelse(df3$DPF == '1', 1, 0)
df3 <- df3 %>% select(-c(Location, Character, Intensity, Nausea, Phonophobia, Photophobia, Dysphasia, Dysarthria, 
                         Vertigo, Vomit, Tinnitus, Hypoacusis, Diplopia, Defect, Conscience, Paresthesia, DPF))

str(df3)

set.seed(2)
nn = neuralnet(Type_num~., data=df3[train,], hidden=c(10,5), linear.output = FALSE)
plot(nn, rep = "best")

# test the model
pred <- predict(nn, df3[-train,])
yhat.nn <- as.factor(max.col(pred))
levels(yhat.nn)

t.nn <- confusionMatrix(yhat.nn, df.test)
t.nn$table
error <- mean(yhat.nn != df.test)
print(paste('Accuracy =', 1-error))

#############################################################################################################
#                                         3. Boosting                                                       #
#############################################################################################################

# boosting
set.seed(123)

adaboost <- boosting(Type_num~., data=df2[train,], boos=TRUE)

# test the model
yhat.boost = predict(adaboost, newdata = df2[-train ,], n.trees=500, type = 'response')
p.yhat.boost <- as.factor(apply(yhat.boost$prob, 1, which.max))

t.boost <- confusionMatrix(p.yhat.boost, df.test)
t.boost$table
error <- mean(p.yhat.boost != df.test)
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
svm.tune.out = tune(svm, Type_num~., data = df2[train,], kernel ="linear", 
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
svm.tune.out2 = tune(svm, Type_num~., data = df2[train,], kernel ="polynomial", 
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
svm.tune.out3 = tune(svm, Type_num~., data = df2[train,], kernel ="radial", 
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

# to choose optimal k

ctrl <- trainControl(method="repeatedcv", repeats = 3)
knn.fit <- train(Type_num ~ ., data = df2[train,], method = "knn", 
                 trControl = ctrl, tuneGrid = expand.grid(k = seq(1,10,1)))
knn.fit

#Use plots to see optimal number of clusters:
#Plotting yields Number of Neighbors Vs accuracy (based on repeated cross validation)
plot(knn.fit)

# use k = 1

knn.df1 <- knn(train = df2[train,], test = df2[-train,],
                      cl = df2[train,]$Type_num, k = 1)

t.knn <- confusionMatrix(knn.df1, df.test)
t.knn$table
error <- mean(knn.df1 != df.test)
print(paste('Accuracy =', 1-error))

# use k = 2

knn.df2 <- knn(train = df2[train,], test = df2[-train,],
               cl = df2[train,]$Type_num, k = 2)

t.knn <- confusionMatrix(knn.df2, df.test)
t.knn$table
error <- mean(knn.df2 != df.test)
print(paste('Accuracy =', 1-error))

# use k = 3

knn.df3 <- knn(train = df2[train,], test = df2[-train,],
               cl = df2[train,]$Type_num, k = 3)

t.knn <- confusionMatrix(knn.df3, df.test)
t.knn$table
error <- mean(knn.df3 != df.test)
print(paste('Accuracy =', 1-error))

# use k = 4

knn.df4 <- knn(train = df2[train,], test = df2[-train,],
               cl = df2[train,]$Type_num, k = 4)

t.knn <- confusionMatrix(knn.df4, df.test)
t.knn$table
error <- mean(knn.df4 != df.test)
print(paste('Accuracy =', 1-error))

# use k = 5

knn.df5 <- knn(train = df2[train,], test = df2[-train,],
                      cl = df2[train,]$Type_num, k = 5)

t.knn <- confusionMatrix(knn.df5, df.test)
t.knn$table
error <- mean(knn.df5 != df.test)
print(paste('Accuracy =', 1-error))


