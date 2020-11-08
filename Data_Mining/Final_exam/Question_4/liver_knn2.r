train <- read.csv("V:/DataScience_2019501043/Data_Mining/Final_exam/Question_4/liver_train.csv")
test <- read.csv("V:/DataScience_2019501043/Data_Mining/Final_exam/Question_4/liver_test.csv")


##extract 7th column of train dataset because it will be used as 'cl' argument in knn function.
data_target_category <- train[,7]
##extract 7th column if test dataset to measure the accuracy
data_test_category <- test[,7]
library(class)
##run knn function
anyNA(data_target_category)

perform_knn <- function(n) {
  pr <- knn(train,test,cl=data_target_category,k=n)
  ##create confusion matrix
  tab <- table(pr,data_test_category)
  print(tab)
  ##this function divides the correct predictions by total number of predictions that tell us how accurate teh model is.
  
  accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
  accuracy(tab)
}
perform_knn(1)
perform_knn(2)
perform_knn(3)

