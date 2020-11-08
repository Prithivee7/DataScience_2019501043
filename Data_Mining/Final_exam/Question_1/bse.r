bse_data <- read.csv("BSE_Sensex_Index.csv",header=FALSE)
summary(bse_data)
colnames(bse_data)
#to get attribute names 
attribute_names <- names(bse_data)
attribute_names

number_of_rows <- nrow(bse_data)
number_of_rows
#15446 rows are done then the last row is done uniquely
open_type <- c()
for(i in 1:15446) {
  open_type[i] <- (bse_data$Open[i] - bse_data$Open[i+1]) / (bse_data$Open[i+1])
}
open_type[15447] <- (open_type[15444] + open_type[15445] + open_type[15446]) / 3
open_type
bse_data$open_type <- open_type
bse_data$open_type

high_type <- c()
for(i in 1:15446) {
  high_type[i] <- (bse_data$High[i] - bse_data$High[i+1]) / (bse_data$High[i+1])
}
high_type[15447] <- (high_type[15444] + high_type[15445] + high_type[15446]) / 3
high_type
bse_data$high_type <- high_type
bse_data$high_type

low_type <- c()
for(i in 1:15446) {
  low_type[i] <- (bse_data$Low[i] - bse_data$Low[i+1]) / (bse_data$Low[i+1])
}
low_type[15447] <- (low_type[15444] + low_type[15445] + low_type[15446]) / 3
low_type
bse_data$low_type <- low_type
bse_data$low_type

close_type <- c()
for(i in 1:15446) {
  close_type[i] <- (bse_data$Close[i] - bse_data$Close[i+1]) / (bse_data$Close[i+1])
}
close_type[15447] <- (close_type[15444] + close_type[15445] + close_type[15446]) / 3
close_type
bse_data$close_type <- close_type
bse_data$close_type

volume_type <- c()
for(i in 1:15446) {
  volume_type[i] <- (bse_data$Volume[i] - bse_data$Volume[i+1]) / (bse_data$Volume[i+1])
}
volume_type[15447] <- (volume_type[15444] + volume_type[15445] + volume_type[15446]) / 3
Volume_type
bse_data$volume_type <- volume_type
bse_data$volume_type

adjcls_type <- c()
for(i in 1:15446) {
  adjcls_type[i] <- (bse_data$Adj.Close[i] - bse_data$Adj.Close[i+1]) / (bse_data$Adj.Close[i+1])
}
adjcls_type[15447] <- (adjcls_type[15444] + adjcls_type[15445] + adjcls_type[15446]) / 3
adjcls_type
bse_data$adjcls_type <- adjcls_type
bse_data$adjcls_type

#Sample preparation "with replace" for 1000 and 2000
sample1 <- sample_n(bse_data,1000,replace = TRUE)
sample2 <- sample_n(bse_data,3000,replace = TRUE)

#Summary provides median, mean,max,min, 1st and 3rd quantile
summary(sample1)
summary(sample2)

#As we can se we got more or less similar data with the population also
# Mean and median are almost similar
summary(bse_data)

open_vector <- bse_data$Open
high_vector <- bse_data$High
low_vector <- bse_data$Low
close_vector <- bse_data$Close

boxplot(open_vector, high_vector,low_vector,close_vector,
        main = "plot open,high,low and close",
        names = c("Open","High","Low","Close"),
        col = c("orange","red"),
        horizontal = TRUE,
        notch = TRUE)

hist(bse_data$Close, 
     ylim = c(0,2000),
     col ="red",
     xlab = "adj cls")