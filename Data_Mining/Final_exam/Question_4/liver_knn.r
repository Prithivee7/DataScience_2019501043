data <- read.csv("V:/DataScience_2019501043/Data_Mining/Final_exam/Question_4/Liver_data.csv",header=FALSE)
n <- nrow(data)
for (row in 1:n) {
  price <- data[row, 6]
  
  if(!is.null(price) & !is.na(price) & length(price) > 0 & price < 5 & price >= 0) {
    data[row,6] <- 1
  } else if(!is.null(price) & !is.na(price) & length(price) > 0 & price < 10 & price >= 5) {
    data[row,6] <- 2
  } else if(!is.null(price) & !is.na(price) & length(price) > 0 & price < 15 & price >= 10) {
    data[row,6] <- 3
  }   else if(!is.null(price) & !is.na(price) & length(price) > 0 & price <= 20 & price >= 15) {
    data[row,6] <- 4
  }
}

data[,6] = as.integer(data[,6])
#head(data)

##Generate a random number that is 90% of the total number of rows in dataset.
ran <- sample(1:nrow(data), 0.9 * nrow(data)) 
##the normalization function is created
nor <-function(x) { (x -min(x))/(max(x)-min(x))   }

##Run nomalization on first 6 coulumns of dataset because they are the predictors
data_norm <- as.data.frame(lapply(data[,c(1,2,3,4,5,6)], nor))
#is.na(data_norm)

#summary(data_norm)

data_train <- data_norm[ran,] 
##extract testing set
data_test <- data_norm[-ran,] 
##extract 7th column of train dataset because it will be used as 'cl' argument in knn function.
data_target_category <- data[ran,7]
##extract 7th column if test dataset to measure the accuracy
data_test_category <- data[-ran,7]
##load the package class
library(class)
##run knn function
anyNA(data_target_category)

perform_knn <- function(n) {
  pr <- knn(data_train,data_test,cl=data_target_category,k=n)
  ##create confusion matrix
  tab <- table(pr,data_test_category)
  print(tab)
  ##this function divides the correct predictions by total number of predictions that tell us how accurate teh model is.
  
  accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
  print(accuracy(tab))
  mis <- 1 - (accuracy(tab)*0.01)
  print(mis)
}
perform_knn(1)
perform_knn(2)
perform_knn(3)



