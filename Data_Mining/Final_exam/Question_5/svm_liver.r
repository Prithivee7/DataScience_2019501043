#install.packages("caret")
#install.packages("e1071")
data <- read.csv("V:/DataScience_2019501043/Data_Mining/Final_exam/Question_4/Liver_data.csv",header=FALSE)
data
str(data)
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

print(data[,7])
data[,6] = as.integer(data[,6])
intrain <- createDataPartition(y = data$V7, p= 0.7, list = FALSE)
training <- data[intrain,]
testing <- data[-intrain,]
training
testing

dim(training) 
dim(testing)

anyNA(data)
training[["V7"]] = factor(training[["V7"]])
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
svm_Linear <- train(V7 ~., data = training, method = "svmLinear",
                    trControl=trctrl,
                    preProcess = c("center", "scale"),
                    tuneLength = 10)

svm_Linear
#We get an accuracy score = 0.6884 and kappa = 0.30145
#Mis = 1 - 0.6884 = 0.3116
test_pred <- predict(svm_Linear, newdata = testing)
#test_pred

