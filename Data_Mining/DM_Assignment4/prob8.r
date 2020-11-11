train <-read.csv("V:/DataScience_2019501043/Data_Mining/DM_Assignment4/sonar_train.csv",header=F)
test <-read.csv("V:/DataScience_2019501043/Data_Mining/DM_Assignment4/sonar_test.csv",header=F)

k <- test[,1:2]
plot(k,pch=19,xlab=expression(k[1]), ylab=expression(k[2]))

kmeanfit<-kmeans(k,2)
points(kmeanfit$centers,pch=19,col="red",cex=2)

library(class)
knnfit<-knn(kmeanfit$centers,k,as.factor(c(-1,1)))
points(k,col=1+1*as.numeric(knnfit),pch=19)


y<-test[,61]
1-sum(knnfit==y)/length(y)

#mis classification error is 0.4805195
#0.525641

#8b) We can choose any of value as K but if we choose the K value as  odd 
#then we get ambiguity  and the error may increase. So we should have the
#odd number of clusters for even number of class in order to reduce the
#ambiguity in making the decision for the clusters