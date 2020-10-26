data = read.csv("V:/DataScience_2019501043/Data_Mining/DM_Assignment2/twomillion.csv",header=FALSE)
sample_Replace <- sample(data$V1, 10000, replace = TRUE)

calculate <- function(data_Set) {
  avg = mean(data_Set)
  maxi <- max(data_Set)
  variance <- var(data_Set)
  quart1 <- quantile(data_Set,0.25)
  print(sprintf("Mean : %f, max : %f, variance : %f, 1st quartile : %f",avg,maxi,variance,quart1))
}

calculate(sample_Replace)
calculate(data$V1)
