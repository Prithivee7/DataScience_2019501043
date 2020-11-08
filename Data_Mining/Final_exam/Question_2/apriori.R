#install.packages("arules")
data <-read.transactions(file = "V:/DataScience_2019501043/Data_Mining/Final_exam/Question_2/aprori.csv",sep=",")
data
#support = How many times to (A,B) occur in all the 10 rows
#confidence = If(A,B) occur what is the probability to get C
rules <- apriori(data,  
                 parameter = list(support = 0.01, confidence = 0.2))
#rules <- apriori(data)
inspect(rules[1:80])

rules_by_lift <- sort(rules, by = "count")
#We sort it and we display till 3. Thats y 21
inspect(rules_by_lift[1:21])
