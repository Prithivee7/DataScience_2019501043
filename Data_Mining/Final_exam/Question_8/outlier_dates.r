library(dplyr)
data <-read.csv("V:/DataScience_2019501043/Data_Mining/Final_exam/Question_1/bse_index.csv",header=TRUE)
close <- data[['Close']]

cal_close <- function(close){
  y <- numeric()
  for (i in 1:(length(close)-1)){
    y <- c(y, (close[i]-close[i+1])/close[i+1])
  }
  y
}

res_vector <- cal_close(close)
#For last value we find the avg of 3 penultimate rows
sum_3_rows <-  res_vector[length(res_vector)] + res_vector[length(res_vector)-1] + res_vector[length(res_vector)-2]
lastvalue <- (sum_3_rows)/3
res_vector <- append(res_vector,lastvalue)
mean <- mean(res_vector)
std <- sd(res_vector)

z_score <- function(res_vector){
  z <- numeric()
  for (i in 1:length(res_vector)){
    z <- c(z, (res_vector[i]-mean)/std)
  }
  z
}

z <- z_score(res_vector)
data['Close'] = z
#We filter based on -3 and +3 z scores
outlier_years <- filter(data, Close <-3 & Close > 3)
outlier_years[,1]