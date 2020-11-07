train <-read.csv("V:/DataScience_2019501043/Data_Mining/DM_Assignment4/sonar_train.csv",header=F)
test <-read.csv("V:/DataScience_2019501043/Data_Mining/DM_Assignment4/sonar_test.csv",header=F)

for(i in  1:61) {
  ftp(i)
}

ftp <- function(i) {
  x <- test[,1:2]
  plot(x,pch=19,xlab=expression(x[1]),
       ylab=expression(x[i]))
  #2 in kmeans refers to the number of centers
  fit<-kmeans(x, 2)
  fit
  #pch - plot character determines the shape of the character. In this case 18 is diamond
  #cex - character expansion (numerical vector)
  points(fit$centers,pch=19,col="blue",cex=2)
  
  library(class)
  c(-1,1)
  
  #c combines its variable to form a vector
  #c(-1,1) means the values -1,1. It can also be used to represent 1:10 
  #kNN(form, train, test, norm = T, norm.stats = NULL, ...)
  knnfit<-knn(fit$centers,x,as.factor(c(-1,1)))
  
  #col - colour code
  points(x,col=1+1*as.numeric(knnfit),pch=19)
} 