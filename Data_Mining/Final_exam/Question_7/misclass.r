data <- read.csv("V:/DataScience_2019501043/Data_Mining/Final_exam/Question_4/Liver_data.csv",header=FALSE)
#seed - to randomly assign centroids. k means custering startswith randomly assigning centroids

set.seed(123)
#no of centers = 4
#nstart - how many random sets should be chosen for 4 centers
#iter.max - maximum number of iterations allowed
res.km <- kmeans(scale(data[,-7]), 4, nstart = 25,iter.max=10)
#print(res.km)

perform_kmeans <- function(n) {
  pr <- knn(data_train,data_test,cl=data_target_category,k=n)
  ##create confusion matrix
  tab <- table(pr,data_test_category)
  print(tab)
  ##this function divides the correct predictions by total number of predictions that tell us how accurate teh model is.
  
  accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
  accuracy(tab)
}
perform_kmeans(3)
