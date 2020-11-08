data <- read.csv("V:/DataScience_2019501043/Data_Mining/Final_exam/Question_4/Liver_data.csv")
options(tibble.print_max = Inf)
s <- data[order(data[,7]),]
train <- s[c(1:144),c(1:7)]
test <- s[c(145:345),c(1:7)]
colnames(train) <- c("mcv","alkphos","sgpt","sgot","gammagt","drinks","selector")
colnames(test) <- c("mcv","alkphos","sgpt","sgot","gammagt","drinks","selector")

n <- nrow(train)
for (row in 1:n) {
  price <- train[row, "drinks"]
  if(!is.null(price) & !is.na(price) & length(price) > 0 & price < 5 & price >= 0) {
    train[row,"drinks"] <- 1
  } else if(!is.null(price) & !is.na(price) & length(price) > 0 & price < 10 & price >= 5) {
    train[row,"drinks"] <- 2
  } else if(!is.null(price) & !is.na(price) & length(price) > 0 & price < 15 & price >= 10) {
    train[row,"drinks"] <- 3
  }   else if(!is.null(price) & !is.na(price) & length(price) > 0 & price <= 20 & price >= 15) {
    train[row,"drinks"] <- 4
  }
}

k <- nrow(test)
for (row in 1:k) {
  price <- test[row, "drinks"]
  if(!is.na(price) & price < 5 & price >= 0) {
    test[row,"drinks"] <- 1
  } else if(!is.na(price) & price < 10 & price >= 5) {
    test[row,"drinks"] <- 2
  }   else if(!is.na(price) & price < 15 & price >= 10) {
    test[row,"drinks"] <- 3
  }   else if(!is.na(price) & price <= 20 & price >= 15) {
    test[row,"drinks"] <- 4
  }
}


write.csv(train,"V:/DataScience_2019501043/Data_Mining/Final_exam/Question_4/liver_train.csv", row.names = FALSE)
write.csv(test,"V:/DataScience_2019501043/Data_Mining/Final_exam/Question_4/liver_test.csv", row.names = FALSE)

#Before running this code remove the last line from test dataset because it has NA

