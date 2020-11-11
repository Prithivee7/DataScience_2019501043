train <-read.csv("V:/DataScience_2019501043/Data_Mining/DM_Assignment4/sonar_train.csv",header=F)
test <-read.csv("V:/DataScience_2019501043/Data_Mining/DM_Assignment4/sonar_test.csv",header=F)

# Q2. *******************************************
#Use Kmeans() with all `the default values to find the k=2 solution for the
#first two columns of the sonar test data. Plot these two columns. Also plot
#the fitted cluster centers using a different color. Finally use the knn() 
#function to assign the cluster membership for the points to the nearest
#cluster center. Color the points according to their cluster membership.
#Show your R commands for doing so.

#for only first two columns
k <- test[,1:2]
plot(k,pch=19,xlab=expression(k[1]), ylab=expression(k[2]))

kmeanfit<-kmeans(k,2)
points(kmeanfit$centers,pch=19,col="red",cex=2)

library(class)
knnfit<-knn(kmeanfit$centers,k,as.factor(c(-1,1)))
points(k,col=1+1*as.numeric(knnfit),pch=19)

# Q3. ********************************
#Graphically compare the cluster memberships from the previous problem to the
#actual labels in the test data. Also compute the misclassification error
#that would result if you used your clustering rule to classify the data.
#Show your R commands for doing so.

y<-test[,61]
1-sum(knnfit==y)/length(y)

#mis classification error is 0.4805195
#0.525641

#Q4) *********************************
#Repeat the previous problem using all 60 columns. Show your R 
#commands for doing so.

#for all 60 columns
kAll<-test[,1:60]
#plot(kAll,pch=19,xlab=expression(kAll[1]), ylab=expression(kAll[2]))

kmeanfit<-kmeans(kAll,2)
points(kmeanfit$centers,pch=19,col="red",cex=2)

library(class)
knnfit<-knn(kmeanfit$centers,kAll,as.factor(c(-1,1)))
points(kAll,col=1+1*as.numeric(knnfit),pch=19)

y <- test[,61]
1-sum(knnfit==y)/length(y)
#missclassification error is 0.454545
#0.4358974

#Q6. ****************************
#Repeat the previous problem by writing a loop and verify that the final 
#answer is the same and show your R commands for doing so.
x<-c(1,2,2.5,3,3.5,4,4.5,5,7,8,8.5,9,9.5,10) 

center1<-1
center2<-2

for (k in 2:10){
  cluster1<-x[abs(x-center1[k-1])<=abs(x-center2[k-1])]
  cluster2<-x[abs(x-center1[k-1])>abs(x-center2[k-1])]
  center1[k]<-mean(cluster1)
  center2[k]<-mean(cluster2)
}
cluster1
cluster2
#Q7. ****************************
#Verify that the kmeans function gives the same solution for the previous
#problem when you use all of the default values and show your R commands 
#for doing so.


kmeans(x,2)
# Verified, m1 = 3.187500, m2 = 8.66667


#Q8. ****************************
#Euclidean distance
x1<-c(1,2)
x2<-c(5,10)

data<-matrix(c(x1,x2),nrow=2,byrow=T)
dist(data)


#Q9. ****************************
#Euclidean distance
x1<-c(1,2,3,6) 
x2<-c(5,10,4,12)
data<-matrix(c(x1,x2),nrow=2,byrow=T)
dist(data)
#ans:10.81665

#Q11. ****************************
#Use a z score cut off of 3 to identify any outliers using the grades for
#the first midterm at www.stats202.com/spring2008exams.csv. Are there any
#outliers according to the z=+/-3 rule? What is the value of the largest
#z score and what is the value of the smallest (most negative) z score?
#Show your R commands.

data <- read.csv("V:/DataScience_2019501043/Data_Mining/DM_Assignment5/spring2008exams.csv")
exam2mean<-mean(data[,2], na.rm=TRUE) 
exam2mean
exam2sd<-sd(data[,2],na.rm=TRUE)
z<-(data[,2] - exam2mean)/exam2sd
sort(z)

#largest =  1.84957968, smallest = -2.28375331


#Q12. ****************************
#Use a z score cut off of 3 to identify any outliers using the grades for
#the second midterm at www.stats202.com/spring2008exams.csv. Are there any
#outliers according to the z=+/-3 rule? What is the value of the largest
#z score and what is the value of the smallest (most negative) z score?
#Show your R commands.
data <- read.csv("V:/DataScience_2019501043/Data_Mining/DM_Assignment5/spring2008exams.csv")
exam2mean<-mean(data[,3], na.rm=TRUE) 
exam2mean
exam2sd<-sd(data[,3],na.rm=TRUE)
z<-(data[,3] - exam2mean)/exam2sd
sort(z)
#largest = 1.29972622, smallest = -2.39622252

#Q14. ****************************
#Identify any outliers more than 1.5 IQR's above the 3rd quartile or below
#the 1st quartile. Verify that these are the same outliers found by the 
#boxplot function using the grades for the second midterm at 
#www.stats202.com/spring2008exams.csv. Show your R commands and include the 
#boxplot. Are any of the grades for the second midterm outliers by this rule?
#If so, which ones?
data <- read.csv("V:/DataScience_2019501043/Data_Mining/DM_Assignment5/spring2008exams.csv")
#1st quantile
q1 = quantile(data$Midterm.2, .25, na.rm = TRUE)
#2nd quantile
q3 = quantile(data$Midterm.2, .75, na.rm = TRUE)
interQrtile <- q3 - q1
interQrtile

data[(data$Midterm.2 > q3 + 1.5 * interQrtile), 3]
data[(data$Midterm.2 > q1 - 1.5 * interQrtile), 3]

boxplot(data$Midterm.2, col="Green", main="Exam Scores", xlab=("Exam 2"),ylab="Exam Score")

#Q15. ****************************
#Use functions to fit a least squares regression model which predicts the
#exam 2 score as a function of the exam 1 score for the data 
#spring2008exams.csv. Plot the fitted line and determine for which points
#the fitted exam 2 values are the furthest from the actual values using the 
#model residuals using the midterm grades at www.stats202.com/spring2008exams.csv. 
#Be sure to include the plot. Which student had the largest 
#POSITIVE residual? Show your R commands.

data <- read.csv("V:/DataScience_2019501043/Data_Mining/DM_Assignment5/spring2008exams.csv")
model <- lm(data$Midterm.2 ~ data$Midterm.1)
plot(data$Midterm.1, data$Midterm.2, pch=19,xlab="Exam 1", ylab="Exam2",xlim=c(100,200),ylim=c(100,200))
abline(model)
sort(model$residuals)
#Largest  positive residual: 18.1717673

