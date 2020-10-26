ohio_data <-read.csv("V:/DataScience_2019501043/Data_Mining/DM_Assignment2/OH_House_Prices.csv",header=FALSE) 

mean(ohio_data[,1])
median(ohio_data[,1])

#mean is greater than the median
plot(ohio_data[,1])

install.packages("e1071")
hist(ohio_data[,1])
skewness(ohio_data[,1])
#positively skewed
median(ohio_data[,1])
#The median changes by the scale in which we change it
median(ohio_data[,1]+10)
median(ohio_data[,1]*2)

ages<-c(19,23,30,30,45,25,24,20)
sd(ages)
#If we add constant values to every entry then sd remains unchanged
#But multiplying by a factor increases the sd by that factor
sd(ages+10)
sd(ages*2)
