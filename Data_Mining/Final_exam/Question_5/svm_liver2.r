train <- read.csv("V:/DataScience_2019501043/Data_Mining/Final_exam/Question_4/liver_train.csv")
test <- read.csv("V:/DataScience_2019501043/Data_Mining/Final_exam/Question_4/liver_test.csv")

anyNA(train)
anyNA(test)
train
train[["V7"]] = factor(train[["X1.V7"]])
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
svm_Linear <- train(X1.V7 ~., data = train, method = "svmLinear",
                    trControl=trctrl,
                    preProcess = c("center", "scale"),
                    tuneLength = 10)

#Since our training set has only one level "1" we have to drop the column 


