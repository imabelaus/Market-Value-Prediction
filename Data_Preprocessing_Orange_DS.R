# Imanol's Orange Code 


train_X <- read.delim("~/Downloads/assignment2_data/train_X.csv", header=FALSE)
View(train_X)
train_X[train_X == ""] = NA
View(train_X)


# lets check the data
summary(train_X)
dim(train_X)

non_empty_column_values <- vector()
non_empty_column_values_percent <- vector()
unique_column_values <- vector()

for (i in 1:ncol(train_X)){
  non_empty_column_values[i] <- sum(!is.na(train_X[,i]))
  non_empty_column_values_percent[i] <- (non_empty_column_values[i]/nrow(train_X))*100
  unique_column_values[i] <- length(unique(train_X[,i]))
}

#creates a descending sorted list of the non-empty value percentages
# importantly this sorting keeps the original index values which represent the column index (or name) 
sorted_by_non_empty <- sort(non_empty_column_values_percent, decreasing = TRUE, index.return = TRUE)
print(sorted_by_non_empty$x) #prints out the percentages
print(sorted_by_non_empty$ix)

# too many incomplete values. There is at least 5 columns with no values at all.
# removing all columns with more than 10% of NA/NULLS

miss <- c()
for(i in 1:ncol(train_X)) {
if(length(which(is.na(train_X[,i]))) > 0.3*nrow(train_X)) miss <- append(miss,i)
}

# we will call that data2

data2 <- train_X[,-miss]
dim(data2)
View(data2)




# Our data has numeric and non numeric values.
# Out of data 2, we will see how many columns are numeric and which are non numeric.

numeric_data <- data2[sapply(data2,is.numeric)]
dim(numeric_data)



factor_data <- data2[sapply(data2,is.factor)]
dim(factor_data)
head(factor_data)


# Here we see the unique values of the factor data

summary(factor_data)
apply(factor_data,2, function(x) length(unique(x)))



# Lets get rid of high dimensionality data (with more than 100 unique values)

too_large <- c()
for(i in 1:ncol(factor_data)){
if(length(unique(factor_data[,i])) > 20) too_large <- append(too_large,i)
  }
factor_data1 <- factor_data[,-too_large]
dim(factor_data1)
View(factor_data1)

apply(factor_data1,2, function(x) length(unique(x)))


#OUR DATA SO FAR HAS THIS DIMENSIONS

# factor data
dim(factor_data1)


# numeric data
dim(numeric_data)


#to do - imputation in numeric data

--- 
  
# Later 

# outlier detection.
# correlated columns
# analizing factor data (¿imputation?¿removal? --> ASK DANIEL)








