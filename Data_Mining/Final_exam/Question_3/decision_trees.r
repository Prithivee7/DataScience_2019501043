
#install.packages("partykit")
#install.packages("rplot.plot")
data <- read.csv("V:/DataScience_2019501043/Data_Mining/Final_exam/Question_3/lenses.data.csv",header=FALSE)
#to remove the first unwanted columns

#method_1
data <- data[-1]
colnames(data) <- c("AGE", "SPECTACLE", "ASTIGMATIC", "TEARS", "LENS")
lens_tree = rpart(LENS ~  AGE + SPECTACLE + ASTIGMATIC + TEARS, data=data)
library(rplot.plot)
rpart.plot(lens_tree, nn=TRUE)
prp(lens_tree)


#method_2
y<-as.factor(lensdata[,5])
x<-lensdata[,1:4]

model1<-rpart(y~.,x, parms = list(split = 'information'),
              control=rpart.control(minsplit=0,minbucket=0, xval=0,maxdepth=5))


library(rpart.plot)
rpart.plot(model1)



#Calculating gain and error_rate
gain <- sum(y==predict(model1,x,type="class"))/length(y)
gain
error_rate <- 1-sum(y==predict(model1,x,type="class"))/length(y)
error_rate


