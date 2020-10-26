install.packages("dplyr")
football <- read.csv("V:/DataScience_2019501043/Data_Mining/DM_Assignment2/football.csv",header=FALSE)
plot(football[,2],football[,3],xlim=c(0,12),ylim=c(0,12),pch=15,col="blue",xlab="2003 Wins",ylab="2004 Wins",
      main="Football Wins (Prithivee)")
abline(c(0,1))


col1 <- tail(football[,2],-1)
col1 <- strtoi(col1)
col2 <- tail(football[,3],-1)
col2 <- strtoi(col2)

correlation <- function(t1,t2) {
  cor(t1, t2,  method = "pearson")  
}

correlation(col1,col2)
correlation(col1+10,col2)
correlation(col1,col2+10)
correlation(col1,col2*2)
# We get a correlation of 0.654. It stays the same for all the cases.





football[,c(2,3)]
