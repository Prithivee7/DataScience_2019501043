ohio_data <-read.csv("V:/DataScience_2019501043/Data_Mining/DM_Assignment2/OH_House_Prices.csv",header=FALSE)
cal_data <-read.csv("V:/DataScience_2019501043/Data_Mining/DM_Assignment2/CA_House_Prices.csv",header=FALSE)

boxplot(cal_data[,1],ohio_data[,1],col="blue",main="House Boxplot",names=c("CA houses", "Ohio houses"),ylab="Prices(in thousands)")
hist(cal_data[,1]*1000,breaks=seq(0,3500000,by=500000),col="red",xlab="Prices",ylab="Frequency",main="Prithivee's CA House Plot")


#ECDF - Empirical cumulative distribution function
plot(ecdf(cal_data[,1]),verticals= TRUE,do.p = FALSE,main ="ECDF for House Prices (Prithivee)",xlab="Prices(in thousands)",ylab="Frequency")
lines(ecdf(ohio_data[,1]),verticals= TRUE,do.p = FALSE,col.h="red",col.v="red",lwd=4)
legend(2100,.6,c("CA Houses","OH Houses"), col=c("black","red"),lwd=c(1,4)
