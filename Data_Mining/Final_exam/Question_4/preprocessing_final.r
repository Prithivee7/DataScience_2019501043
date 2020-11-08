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


data_split <- split(data,data$V7)
train <- data_split[1]
test <- data_split[2]

train
test
write.csv(train,"V:/DataScience_2019501043/Data_Mining/Final_exam/Question_4/liver_train.csv", row.names = FALSE)
write.csv(test,"V:/DataScience_2019501043/Data_Mining/Final_exam/Question_4/liver_test.csv", row.names = FALSE)