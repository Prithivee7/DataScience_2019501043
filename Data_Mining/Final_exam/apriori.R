install.packages("arules")
data <-read.csv("V:/DataScience_2019501043/Data_Mining/Final_exam/aprori.csv")
data
rules <- apriori(data,  
                 parameter = list(supp = 0.01, conf = 0.2))
inspect(rules[1:10]) 
