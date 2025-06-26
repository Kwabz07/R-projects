# set wokring directory
#setwd("C:\\Users\\KNUST\\Documents")
(aaa = Sys.time())
#upload all neccessary packages
library(tidyverse)
library(caret)
library(multiROC)
library(MASS)
library(pROC)
library(mda)
library(klaR)
theme_set(theme_classic())
n = 100

KNNdata = matrix(NA,nrow = n, ncol = 4)
colnames = c("ID","AUC","ACU","BCR")
dimnames(KNNdata)[[2]] = colnames

for (i in 1:n) {
#set random number generator
set.seed(i)
#preprocess data
ColumnData=read.csv ("recColData.csv", header =T)  #load rectangular column data
ColumnData[,6]= as.factor(ColumnData[,6])  #label column 6 as a categorical variable
ColumnData[,7]= as.factor(ColumnData[,7]) #label column 7 as a categorical variable
ColumnData[,11]= as.factor(ColumnData[,11]) #label column 11 as a categorical variable
#attach(ColumnData) # allows header names to be used direct in script
train = createDataPartition(ColumnData$Failure, p = 2/3, list = FALSE, times = 1)  #randomly smaple 80% of the data for training
test = (-train) # prepare the index for the test data
trainingData = ColumnData[train,]  #obtain training data
testData = ColumnData[test,] #obtain test data

# Estimate preprocessing parameters
preproc.param =  trainingData %>%  preProcess(method = c("center", "scale")) #center and scale variables
# Transform the data using the estimated parameters
train.transformed = preproc.param %>% predict(trainingData)  #transformed training data
test.transformed = preproc.param %>% predict(testData)   #transformed test data

#linear discriminant analysis

# Fit the model by training a 10-fold cross-validation scenario
train.transformed[,6]=as.numeric(train.transformed[,6])
train.transformed[,7]=as.numeric(train.transformed[,7])
test.transformed[,6]=as.numeric(test.transformed[,6])
test.transformed[,7]=as.numeric(test.transformed[,7])
train.x = train.transformed[,-11]
groupings = train.transformed[,11]
test.x =  test.transformed[,-11]
#lda.model = train(Failure~., data = train.transformed, method = "qda", trControl = trainControl("cv",number = 10))
#lda.model = mda(train.x,grouping = groupings)
lda.model  = try(knn3(train.x, groupings,k = 5),TRUE)
if (class(lda.model) == "try-error") {
KNNdata[i,1] = i
KNNdata[i,2] = 0
KNNdata[i,3] = 0
KNNdata[i,4] = 0
} else {
# Make predictions
#qda.predictions = lda.model %>% predict(test.transformed)
lda.predictions.class = predict(lda.model,test.x,type="class")
#lda.predictions.class <- predict(Failure~.,newdata=test.transformed) #predict classes of data in test or held-out set

#predict posterior probabilties of candidate classes of data in test or held-out set
lda.predictions.prob = predict(lda.model,test.x, type = "prob")

# Model accuracy
mean(lda.predictions.class==test.transformed$Failure)  #acqire accuracy of LDA model

# confusion matrix as proprotions
table(lda.predictions.class,test.transformed$Failure) %>% 
  prop.table() %>% round(digits = 3)

# more on confusion matrix 
confMat = confusionMatrix(lda.predictions.class, test.transformed$Failure)
#confMat$table
#confMat$byClass
#confMat$overall
ACU = confMat$overall[1]
#ROC curve
lda.prediction.probabilities = lda.predictions.prob
lda.res.roc = multiclass.roc(test.transformed$Failure, lda.prediction.probabilities)
AUC = lda.res.roc$auc  #combined auc of model
res = table(lda.predictions.class,test.transformed$Failure)
#print(res)
# Compute Balanced Classification Rate
BCR=mean(c(res[1,1]/sum(res[,1]),res[2,2]/sum(res[,2]),res[3,3]/sum(res[,3])))
#print(BCR)
KNNdata[i,1] = i
KNNdata[i,2] = AUC
KNNdata[i,3] = ACU
KNNdata[i,4] = BCR
}
}

fname = paste(as.character("KNNdata"),".","csv",sep = '')
write.table(KNNdata,fname,sep=",", row.names = FALSE)

bbb = Sys.time()
difftime(bbb,aaa)

# Compute roc

aa12 = lda.res.roc$rocs$'1/2'[[1]] # binary roc with class 1 as control
aa21 =  lda.res.roc$rocs$'1/2'[[2]] # binary roc with class 2 as control
aa13= lda.res.roc$rocs$'1/3'[[1]] # binary roc with class 1 as control
aa31= lda.res.roc$rocs$'1/3'[[2]] # binary roc with class 3 as control
aa23 = lda.res.roc$rocs$'2/3'[[1]] # binary roc with class 2 as control
aa32 = lda.res.roc$rocs$'2/3'[[2]] # binary roc with class 3 as control

plot.roc(aa12, print.auc = TRUE,print.thres = "best")  #plot roc curve for a particular case

