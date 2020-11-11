#install.packages("randomForest")
train <-read.csv("V:/DataScience_2019501043/Data_Mining/DM_Assignment4/sonar_train.csv",header=F)
test <-read.csv("V:/DataScience_2019501043/Data_Mining/DM_Assignment4/sonar_test.csv",header=F)

library(party)
library(randomForest)

#train
# Create the forest.
output.forest <- randomForest(nativeSpeaker ~ age + shoeSize + score, data = readingSkills)

#output.forest <- randomForest(train[V61] ~ train[V1] + train[V2], 
#                              data = train)

# View the forest results.
print(output.forest) 

