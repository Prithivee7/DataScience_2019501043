library(rpart)

train <-read.csv("V:/DataScience_2019501043/Data_Mining/DM_Assignment4/sonar_train.csv",header=F)
test <-read.csv("V:/DataScience_2019501043/Data_Mining/DM_Assignment4/sonar_test.csv",header=F)

train[,61]

#as.factor - Convert numerical variables into categorical variables
y <- as.factor(train[,61])
x <-train[,1:60]

y_test<-as.factor(test[,61])
x_test<-test[,1:60] 

#minbucket - smallest number of observations that are allowed in a terminal node
#minsplit - smallest number of nodes in the parent node that could be split further
#maxdepth -  prevents the tree from going past a certain depth
#cp - complexity parameter

errors<-matrix(data=NA,nrow=6,ncol=3)

for(dep in 1:6){
  errors[dep,1]<-dep
  fit<-rpart(y~.,x,control=rpart.control(minsplit=0,minbucket=0,cp=-1,maxcompete=0,
                                         xval=0,maxdepth=dep,)) 
  
  errors[dep,2]<-sum(y!=predict(fit,x,type="class"))/length(y)
  errors[dep,3]<-sum(y_test!=predict(fit,x_test,
                                     type="class"))/length(y_test)
}


overfit.model <- rpart(y~., data = train,
                       maxdepth= 5, minsplit=2,
                       minbucket = 1)


rpart(y~., data, parms=list(split=c("information","gini")),
      cp = 0.01, minsplit=20, minbucket=7, maxdepth=30)







minVal<-errors[which.min(errors[,3]),3]
minDep<-errors[which.min(errors[,3]),1]
print(c("Lowest test error of ",round(minVal,3),"and tree depth of ",minDep))