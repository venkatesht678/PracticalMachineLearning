---
title: "Practical Machine Learning"
author: "Venkatesh"
date: "Sunday, Feb 22, 2015"
output:
  html_document:
    theme: spacelab
---
# Background Summary 
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In the project, the goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 

### Libraries
The libraries that are used in this code.

```r
library(caret)
library(kernlab)
library(corrplot)
library(randomForest)
library(knitr)
```


### Loading data and preprocessing the data
Two csv files that contains test and training data were downloaded into a csvdata folder in the current directory. 


```r
# a csvdata folder is created if the one doesnot exits
if (!file.exists("csvdata")) {dir.create("csvdata")}

# file URL and destination file
f1 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
d1 <- "./csvdata/pml-training.csv"
f2 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
d2 <- "./csvdata/pml-testing.csv"

# To download two csv files
download.file(f1, destfile = d1)
download.file(f2, destfile = d2)
dateDownloaded <- date()
```

The training data is loaded into R using the below command.


```r
# read the csv file for training 
training_data<- read.csv("./csvdata/pml-training.csv", na.strings= c("NA",""," "))
```

The NA values which creates noises were removed at first and also the identifiers for the experiment were also removed by removing the first eight columns.


```r
# The data is cleaned by removing NAs
training_data_NAs <- apply(training_data, 2, function(x) {sum(is.na(x))})
training_data_clean <- training_data[,which(training_data_NAs == 0)]

# The identifiers such as name, timestamps were removed
training_data_clean <- training_data_clean[8:length(training_data_clean)]
```

### Here comes the model creation
Now the test data is now splitted in the ratio 70:30 which contains training and cross validation sets for training the model and testing against it.


```r
# This splits the testing data into the ratio of 70:30
inTrain <- createDataPartition(y = training_data_clean$classe, p = 0.7, list = FALSE)
training <- training_data_clean[inTrain, ]
crossval <- training_data_clean[-inTrain, ]
```

To predict classification random forest has been selected. To see the relationship strength among variable a correalation plot was produced.


```r
# plot a correlation matrix
correlMatrix <- cor(training[, -length(training)])
corrplot(correlMatrix, order = "FPC", method = "square", type = "lower", tl.cex = 0.8,  tl.col = rgb(0, 0, 0))
```

![plot of chunk unnamed-chunk-6](figure/unnamed-chunk-6-1.png) 
Figure: correlation plot.


```r
model <- randomForest(classe ~ ., data = training)
model
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = training) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.55%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3903    3    0    0    0 0.0007680492
## B   14 2640    4    0    0 0.0067720090
## C    0   12 2380    3    1 0.0066777963
## D    0    0   26 2224    2 0.0124333925
## E    0    0    4    7 2514 0.0043564356
```

The model has a very small Out-Of-Bag error rate of .56%.

### Cross-validation
For determining the accuracy of the model the results were kept in a confusion matrix along with the actual classification.


```r
# crossvalidaating the model using 30% of data
predictCrossVal <- predict(model, crossval)
confusionMatrix(crossval$classe, predictCrossVal)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1672    2    0    0    0
##          B    2 1132    4    1    0
##          C    0    4 1021    1    0
##          D    0    0   14  950    0
##          E    0    0    0    2 1080
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9949          
##                  95% CI : (0.9927, 0.9966)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9936          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9988   0.9947   0.9827   0.9958   1.0000
## Specificity            0.9995   0.9985   0.9990   0.9972   0.9996
## Pos Pred Value         0.9988   0.9939   0.9951   0.9855   0.9982
## Neg Pred Value         0.9995   0.9987   0.9963   0.9992   1.0000
## Prevalence             0.2845   0.1934   0.1766   0.1621   0.1835
## Detection Rate         0.2841   0.1924   0.1735   0.1614   0.1835
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9992   0.9966   0.9908   0.9965   0.9998
```

The model produced a 99.3% prediction accuracy. Again, the model showed that it is adequeate to predict new data.

### Predictions
To predict the classifications of the 20 results of this new data a separate data set is loaded into R and cleaned for prediction.


```r
# final testing data is also done by the same procedure
test_data <- read.csv("./csvdata/pml-testing.csv", na.strings= c("NA",""," "))
test_data_NAs <- apply(test_data, 2, function(x) {sum(is.na(x))})
test_data_clean <- test_data[,which(test_data_NAs == 0)]
test_data_clean <- test_data_clean[8:length(test_data_clean)]

# To predict classes of test set
predictTest <- predict(model, test_data_clean)
predictTest
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

### Conclusions
It is possible to predict by using lot of information given in the data with a greater accuracy on how a person is preforming an excercise in an  relatively simplified manner. 
