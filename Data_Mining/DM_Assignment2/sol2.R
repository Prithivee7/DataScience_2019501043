data<-read.csv("V:/DataScience_2019501043/Data_Mining/DM_Assignment2/myfirstdata.csv",header=FALSE) 
data

#To know whether a variable is categorical or numerical
col_1 <-is.numeric(data[,1])
col_2 <-is.numeric(data[,2])

col1 <- data[,1]
col2 <- data[,2]

#One of the values is a string so all other values are converted to a string
plot(data[,1],main="Column 1")
plot(data[,2],main="Column 2")

which.nonnumeric <- function (column) {
  which(is.na(suppressWarnings(as.numeric(as.character(column)))))
}

#Used to list the vector names
names(data)

for (name in names(data)) {
  c <- data[[name]]
  r <- which.nonnumeric(c)
  v <- c[r]
  msg <- ''
  if (length(v)) {
    msg <- sprintf("data$%s is qualitative (%s[%d] == '%s')", name, name, r, as.character(v))
  } else {
    msg <- sprintf("data$%s is quantitive (all rows are numeric)", name)
  }
  print(msg)
  
}

#Adding 10 to a string in excel displays !Value



