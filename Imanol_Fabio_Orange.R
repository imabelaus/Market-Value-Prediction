#-------------------------------- CONTENT:

# STEP 1 - work out columns to keep based on the amount of missing data + detecting and removing newa zero variables
# STEP 2 - Summary of the reduced data set and 
# STEP 3 - Checks to see the reduction has materially impacted the data density
# STEP 4 - Investigate Column Data Variance
# STEP 5 : handles factors with a very high number of levels
# STEP 6 - MISSING VALUES IMPUTATION
# STEP 7 - ADDRESSES THE CLASS IMBALANCE
# STEP 8 - PREDICTORS TRANSFORMATIONS

#--------------------------------
# Last Update : 03/03/2019 , 18.40 pm
# Last changes author : Fabio
### NOTES :
#Main changes:
# Added detection and removal of near zero variables (see line 73) ; Added step 5 (line 147),6 (line 196),7 (line 346),8 (line 468)
# There are some open points to discuss in step 5,6,7,8.
#============================================================================================================================

#clears the workspace
rm(list=ls())

############ STEP 1 - work out columns to keep based on the amount of missing data ###########

# setwd("C:\\Users\\Fabio Caputo\\Desktop\\22-10-2018\\Data Science\\MASTER\\MODULES\\ML\\Labs\\MLSDM 2nd COURSEWORK 2018-19\\assignment2_data")

#My dir
setwd("~/Documents")

#imports the train_X file
train_x <- read.csv("train_X.csv", na.strings=c("", "NA"), header=FALSE, sep="\t")

#initialises vectors 
non_empty_column_values <- vector()
non_empty_column_values_percent <- vector()
unique_column_values <- vector()

#iterates through columns and calculates the 
# 1. the number of non_empty values in the column
# 2. the non-empty values as a percent of the overall column
# 3. the unique content values
for (i in 1:ncol(train_x)){
  non_empty_column_values[i] <- sum(!is.na(train_x[,i]))
  non_empty_column_values_percent[i] <- (non_empty_column_values[i]/nrow(train_x))*100
  unique_column_values[i] <- length(unique(train_x[,i]))
}

#creates a descending sorted list of the non-empty value percentages
# importantly this sorting keeps the original index values which represent the column index (or name) 
sorted_by_non_empty <- sort(non_empty_column_values_percent, decreasing = TRUE, index.return = TRUE)
print(sorted_by_non_empty$x) #prints out the percentages
print(sorted_by_non_empty$ix)#prints out the corresponding column index value 

#generates the list of columns to keep (by column index)
# the list is cut off at the threshold where we have 70% or more non-empty values
data_cutoff <- 70;
columns_to_keep <- (sorted_by_non_empty$ix[1:length(sorted_by_non_empty[sorted_by_non_empty$x > data_cutoff])])
print(columns_to_keep)
length(columns_to_keep)

#creates a filtered training set based on the columns we have decided to keep
new_train_x_1 <- subset(train_x, select = c(columns_to_keep))

print(paste("New column count",ncol(new_train_x_1))) # Fabio's proposed change (12/03/2019) - below the previous version (commented out)..
# print("New column count")
# print(ncol(new_train_x_1))

View(new_train_x_1)

#-- 02/03/2019 (Fabio's proposed change)

# DETECTING AND REMOVING NEAR-ZERO VARIACE VARIABLES

library(caret)

# Note : the nearZeroVar function implements this logic: remove a predictor if:
# a. The fraction of unique values over the sample size is low (say 10%)
# b. The ration of the frequency of the most prevalent value to the frequency of the 2nd most prevalent value is large (say 20)

near_zero_vars_1 <- nearZeroVar(new_train_x_1) # when predictor should be removed, a vector of integerers is returned that indicates which columns should be removed
near_zero_vars_2 <- subset(new_train_x_1, select = c(near_zero_vars_1))
dim(near_zero_vars_2)[2] # =10 variables with near-zero variance
View(near_zero_vars_2) # You might want to check V81..

# filter out the variables with non near-zero variance:
new_train_x_2 <- new_train_x_1[! names(new_train_x_1) %in% names(near_zero_vars_2)]
View(new_train_x_2) 

################### STEP 2 - Summary of the reduced data set #################################

#initialises vectors 
new_non_empty_column_values <- vector()
new_non_empty_column_values_percent <- vector()
new_unique_column_values <- vector()
new_column_types <- vector()
new_empty_column_values <- vector()

#iterates through columns and calculates the 
# 1. the number of non_empty entries
# 2. the non-empty entries as a percent of the overall column
# 3. the unique content values
# 4. the column type
for (i in 1:ncol(new_train_x_2)){
  new_non_empty_column_values[i] <- sum(!is.na(new_train_x_2[,i]))
  new_non_empty_column_values_percent[i] <- (new_non_empty_column_values[i]/nrow(new_train_x_2))*100
  new_unique_column_values[i] <- length(unique(new_train_x_2[,i]))
  new_column_types[i] <- class(new_train_x_2[,i])
  new_empty_column_values[i] <- nrow(new_train_x_2)-new_non_empty_column_values[i]
}

print(new_non_empty_column_values_percent)

print(new_unique_column_values)

################# STEP 3 - Checks to see the reduction has materially impacted the data density ##############


# Investigates the mean of non-emtpy entries each row - we do this to see if we have eliminated
# too much data - use the mean of non-empty entries in rows across the two data sets
row_content <- vector()
new_row_content <- vector()
#calculates the number of non-empty values in each row for the original and reduced data set
for (i in 1:nrow(train_x)){row_content[i] <- sum(!is.na(train_x[i,]))}
for (i in 1:nrow(new_train_x_2)){new_row_content[i] <- sum(!is.na(new_train_x_2[i,]))}

#prints out the mean value of non-empty rows in the original and reduced data set
print("Original data set mean row content")
print(mean(row_content))
print((mean(row_content)/ncol(train_x))*100)
print("Revised data set mean row content")
print(mean(new_row_content))
print((mean(new_row_content)/ncol(new_train_x_2))*100)

# Looks like the data has not been materially impacted


################ STEP 4 - Investigate Column Data Variance ##############################  

#creates a matrix with the the number of unique values in each column, the data type and the 
# total number of missing values - NOTE that the index number is shared with the new_train_x_2 
data_handling <- cbind.data.frame(new_unique_column_values, new_column_types, new_empty_column_values )

View(data_handling) #you can manually sort this in the window that pops up


################ 02/03/2019 (Fabio) STEP 5 : handle factors with a very high number of levels (dummifying factors with a huge amount of levels might be problematic)

library(dplyr)
data_handling %>% group_by(new_column_types) %>% summarise(frequency=n())
#new_column_types frequency
#<fct>                <int>
#  1 factor                  24
#  2 integer                 24
#  3 numeric                  9

#=== Exploring the #levels of the factor variables: MAKE THIS BETTER : Plot a line for all values (use PLOTLY?)

fct_rws <- data_handling %>% filter(new_column_types=='factor') %>% summarise(n()) # filters out factor vars

library(data.table)
setDT(data_handling) # when you wotk with the data.table package sometimes you have to change the format of the dataset first..

factor_levels_1 <- setorder(data_handling, new_unique_column_values)[
  new_column_types=='factor'][
    , 1][, percent:=.I/as.numeric(fct_rws)] # get the cumulative percent

# Now using the dplyr package (you can recognise this from the "%>%"sign, typical of dplyr..)..
factor_levels_2 <- factor_levels_1 %>%
  group_by(new_unique_column_values) %>%
  summarise(`%`=max(percent)) # I am interested in the unique value of  "new_unique_column_values" only..specifcially the max

View(factor_levels_2)


# Plot to visualise it.. (this plot can be done better..)
plot(factor_levels_2$new_unique_column_values,
     factor_levels_2$`%`,
     xlab="#levels",
     ylab="percent",
     main="Factor Variables by Levels")

# Now let us apply the work above and filter factors with more than 20 levels - we should agree on a threshold..

fct_lvl_thr <- 20 # This has to be agreed
setDT(new_train_x_2)
new_train_x_3 <-
  new_train_x_2[,lapply(.SD, function(x) {
    if(is.factor(x) & nlevels(x) >= fct_lvl_thr) return(NULL) 
    else return(x)})] # drops the variable IF is both a factor and has more than 20 levels (unique values)

# NOTE - one justification for doing so might be the fact that we haven't been provided with the labels.
# Had we had the labels, we could have attempted to bin the variables with many factors..but in this case, isn't possible


################ STEP 6 - MISSING VALUES IMPUTATION


#---------------------
# BELOW YOU WILL FIND:
# 1. A plot of the missing values with the "mice" package - looks nice
# 2. imputation of missing values with mean (numeric vars) and mode (factors vars) - the extention to the median for numeric vars is straightforw.
# 3. Imputation of missing values with KNN (not working for factors vars..under investigation)
# 4. Imputation of missing values with OLS (not described in the book..It isn't easy, I am studying it)
#--------------------


install.packages("mice")
library(mice)

md.pattern(new_train_x_3)
# returns a tabular form of missing values present in each variable in a data set

# Plot missing values

install.packages("VIM")
library(VIM)
mice_plot <- aggr(new_train_x_3, col=c('navyblue','yellow'),
                  numbers=TRUE, sortVars=TRUE,
                  labels=names(new_train_x_3), cex.axis=0.7,
                  gap=3, ylab=c("Missing data","Pattern"))


#----------- MEAND/MEDIAN/MODE IMPUTATION OF MISSING VALUES

# mean imputation for numerical variables

new_train_x_3_sim <- # "sim" stands for 'simple imputation'..
  new_train_x_3 

# mean imputation for numerical variables
for (i in names(new_train_x_3_sim[, Filter(is.numeric, .SD)]))
  new_train_x_3_sim[[i]][is.na(new_train_x_3_sim[[i]])] <- mean(new_train_x_3_sim[[i]][!is.na(new_train_x_3_sim[[i]])])

# mode imputation for factor variables
for (j in names(new_train_x_3_sim[, Filter(is.factor, .SD)]))
  new_train_x_3_sim[[j]][is.na(new_train_x_3_sim[[j]])] <- names(sort(-table(new_train_x_3_sim[[j]])))[1]

# Check that there are no missing values
my_sim_test <- lapply(new_train_x_3_sim, function(x) sum(is.na(x))) # Note : sometimes this breaks my laptop and R session..
setDT(my_sim_test)
View(my_sim_test)


#----------- Knn imputation method - N O T   W O R K I N G  for categorical vars! Under investigation..


# Just saving the workspace:
save.image("03-03-2019")
# To load the workspace:
#load("03-03-2019")

#====== 1. "impute.knn" function from the "impute" package - the one mentioned in the APM book

#   1.a) Numerical Variables

require(data.table)
numerical <- new_train_x_3[,sapply(new_train_x_3, is.numeric), with=FALSE] # filter out numeric variables
dim(numerical)

# NOTE : to install the "impute" package you might need, according to your R version, 
# to run the below (in comment, so remove the "#" if you need to run) prior to the install.packages()

!requireNamespace("BiocManager", quiely=TRUE)
install.packages("BiocManager")
BiocManager::install("impute", version="3.8")

install.packages("impute")
library(impute)
myknn <- impute.knn(as.matrix(numerical), k=5)
myknn2 <- myknn$data

# Check results - NOTE : the function below causes my R to crash sometimes..careful then
mytest <- lapply(myknn2, function(x) sum(is.na(x)))
setDT(mytest)
View(mytest) # should be all-zeros..

#  1.b) Factor Variables (NOT WORKING!)

require(data.table)
factors <- new_train_x_3[,sapply(new_train_x_3, is.factor), with=FALSE] # filter out factor variables
dim(factors)

library(impute)
myknn_fct <- impute.knn(as.matrix(factors), k=5)
myknn_fct2 <- myknn_fct$data

setDF(myknn_fct2)
View(myknn_fct2)


#======== 2. Trying with the kNN function from the VIM library (you need to install and call the VIM library..not working either..)
# I have tried to apply the function on the factor variables only (the impute package has worked for the numerical vars - SEE ABOVE..)

library(VIM)
?kNN

myknn_fabio <- kNN(new_train_x_3,variable = names(factors) , k=5)


#======== 3. Trying with knn.impute function from the bnstruct package.. no luck

install.packages("bnstruct")
require(bnstruct)

?knn.impute

factors <- as.matrix(factors)
myknnimp <- knn.impute(factors, k=5, cat.var =1:ncol(factors))



#----------- OLS imputation method for missing values -  Under investigation..

# NOTE - there is not example of imputation of missing values with linear regression in the book

# So far :

# Following this source:
# http://www.stat.columbia.edu/~gelman/arm/missing.pdf
# I run the below, but something is going wrong..

numerical = setDF(numerical) # this was created before, check it (is simply the numerical vars of the data set)

lm.imp1 = lm(V1~., data = numerical, subset=V1>0)
pred.1 = predict(lm.imp1, numerical)
impute = function(a, a.impute){
  ifelse(is.na(a), a.impute, a)
}

V1.imp.1 <- impute(numerical$V1, pred.1)

V1.imp.1 = as.data.frame(V1.imp.1)

View(V1.imp.1)

pred.1 = as.data.frame(pred.1)

# Another source : the "mice" package provide the "mice" function that seems to do the job
# I am studying how it works (I get an error when running the below)..would be heloful to ask Daniel what he uses usually..
# https://datascienceplus.com/imputing-missing-data-with-r-mice-package/

imputed_Data <- mice(numerical, m=5, maxit = 50, method = 'pmm', seed = 500)


################ STEP 7 - ADDRESSES THE CLASS IMBALANCE

train_Y <- read.csv("C:\\Users\\Fabio Caputo\\Desktop\\22-10-2018\\Data Science\\MASTER\\MODULES\\ML\\Labs\\MLSDM 2nd COURSEWORK 2018-19\\assignment2_data\\train_Y.csv", sep="\t")

# There are 3 labels : churn, appetency and upselling

# CHURN LABEL:

dim(new_train_x_3_sim)
new_train_x_4_sim <- new_train_x_3_sim
new_train_x_4_sim$churn <- as.factor(train_Y$churn) # adding churn label to the data set
class(new_train_x_4_sim$churn) # checking that it is a factor variable (needed for the upSample/downSample function in caret)
dim(new_train_x_4_sim) # we should now have 1 more column..
head(new_train_x_4_sim$churn) # let s have a look at some records..

View(new_train_x_4_sim) # we should have kept these variables in the data set since the beginning..otherwise, is we sort, we are fucked..


# UPSAMPLING:

require(caret)
set.seed(1103)
upSampledTrain <- upSample(x=new_train_x_4_sim,
                           y=new_train_x_4_sim$churn,
                           yname="churn"
)

dim(new_train_x_4_sim)
dim(upSampledTrain)
table(upSampledTrain$churn)
View(upSampledTrain)

# IMA -------------------> ReliefF

library(CORElearn)
library(AppliedPredictiveModeling)

reliefValues <- attrEval(churn ~ .,data = upSampledTrain, estimator = "ReliefFequalK", ReliefIterations = 50)
head(reliefValues)
# ¿what estimator to use? infoCore(what="attrEvalReg"). "RReliefFequalK"
#RReliefF algorithm where k nearest instances have equal weight.

reliefValues

sort(reliefValues, decreasing = TRUE)

perm <- permuteRelief(x = upSampledTrain[,-1], y = upSampledTrain$churn, nperm = 500, estimator = "ReliefFequalK", ReliefIterations = 50)

head(perm$permutations)

View(upSampledTrain)

# DOWNSAMPLING: but in this example, we will not have to downsample..since we want to work with as many records as possible when modelling (so is just for practice)

set.seed(1103)
downSampledTrain <- downSample(x=new_train_x_4_sim,
                               y=new_train_x_4_sim$churn,
                               yname="churn"
)

dim(new_train_x_4_sim)
dim(downSampledTrain)
table(downSampledTrain$churn)
View(downSampledTrain)

# SYNTENTIC MINORITY OVERSAMPLING TECHNIQUE (SMOTE)

install.packages("DMwR")
library(DMwR)


#------------------------
help(SMOTE)
#======== DESCRIPTION OF THE SMOTE FUNCTION:
#form	
#A formula describing the prediction problem

#data	
#A data frame containing the original (unbalanced) data set

#perc.over	
#A number that drives the decision of how many extra cases from the minority class are generated (known as over-sampling).

#k	
#A number indicating the number of nearest neighbours that are used to generate the new examples of the minority class.

#perc.under	
#A number that drives the decision of how many extra cases from the majority classes are selected for each case generated
#from the minority class (known as under-sampling)

#learner	
#Optionally you may specify a string with the name of a function that implements a classification algorithm that will be applied
#to the resulting SMOTEd data set (defaults to NULL).
#...	
#In case you specify a learner (parameter learner) you can indicate further arguments that will be used when calling this learner.
#---------------------------

set.seed(1103)
smoteTrain <- SMOTE(form = churn~., data=new_train_x_4_sim, k=5) #, perc.over = 600,perc.under=100
dim(new_train_x_4_sim)
dim(smoteTrain)
table(smoteTrain$churn)


# BOOTSTRAPPING OF BOTH MINORITY AND MAJORITY SAMPLE (p. 427 - at the bottom) - NOTE : this has to be reviewed..but the belw is a good basis

set.seed(1103)

require(dplyr)
# Create an empty data frame of the size of the majority sample (i.e. CHURN=1)
boot_min <- data.frame(matrix(vector(),
                              nrow(new_train_x_4_sim[churn==-1]),
                              ncol(new_train_x_4_sim[churn==-1]),
                              dimnames=list(c(), 
                                            c(names(new_train_x_4_sim[churn==1])))))

dim(boot_min)

for (j in 1:ncol(new_train_x_4_sim)){     # for every column of the data set
  boot_min[j] <- sample_n(select(new_train_x_4_sim[churn==1],j),  # sample to bootstrap from (I used the "frac_n" function from dplyr library with the replacement=TRUE option)
                          nrow(new_train_x_4_sim[churn==-1]),   # lenght of new sample
                          replace = TRUE)}      # replace=TRUE is for bootstrapping


boot_max <- data.frame(matrix(vector(),
                              nrow(new_train_x_4_sim[churn==-1]),
                              ncol(new_train_x_4_sim[churn==-1]),
                              dimnames=list(c(), 
                                            c(names(new_train_x_4_sim[churn==1])))))

dim(boot_max)

for (j in 1:ncol(new_train_x_4_sim)){     # for every column of the data set
  boot_max[j] <- sample_n(select(new_train_x_4_sim[churn==-1],j),  # sample to bootstrap from
                          nrow(new_train_x_4_sim[churn==-1]),   # lenght of new sample
                          replace = TRUE)}      # replace is for bootstrapping


# Finally, append the two sets (i.e. churn=1 and churn=-1):
bootTrain <- rbind(boot_min, boot_max)


################ STEP 8 - PREDICTORS TRANSFORMATIONS

# Firstly, check the distribution of the variables - we might eventually need to perform transaformations

# Rule of thumb : if |skewness| > 1.5 (=0 in a normal distrib.) or |kurtosis| > 9 (=3 in a normal distr.) then the distribution is highly skewed..

# Filter out numeric variables:
require(data.table)
setDT(upSampledTrain)
skw_1 <- upSampledTrain[, sapply(upSampledTrain, is.numeric), with=FALSE]

# Check # numeric variables with positive value

all_pst_num <- skw_1[, lapply(.SD,
                              function(x){
                                if(min(x, na.rm=T)>0)
                                  return(x) else return(NULL)})]

dim(all_pst_num)[2] # only 3 numeric variables out of 33 do not include zero or negative values.


# Filter out those where values > 0; and distribution is probably skewed:
require(e1071)
skw_2 <- skw_1[, lapply(.SD,
                        function(x){
                          if(abs(max(x))/abs(min(x))>20 |
                             abs(skewness(x,na.rm = T))>1.5 | abs(kurtosis(x, na.rm=T)) > 9)
                            return(x) else return(NULL)})]


print(paste("Number Variables with high skewness and/or kurtosis :",(dim(skw_2)[2])))

# storing name of variables that are highly skewed
hig_skw_var <- unique(names(skw_2))

require(tidyr)
require(ggplot2)
options(scipen=999)
ggplot(gather(skw_2), aes(value)) +
  geom_histogram() +
  facet_wrap(~key, scales="free_x")+
  ggtitle(expression(bold("Highly Skewed Variables' Distribution")))


# Transform numeric variables so that all of them are positive - and so, sqrt, log, 1/x, box-cox transformations are possible
# NOTE - an alternative that we ought to try is the Yeo-Jonhson transofrmation(which apparently is
# an extension of the box-cox transformation to allow to transoform when dealing with zero or negative values)
# The application seems to be straightforward (is simply an option of the caret preProcess function).
# Hiwever, I am in the stage of understanding it (we could ask Daniel as well to savr time..)

# We need to add a constant to all variables. This constant has to be selected so that all negative and zero values become >0.
# Hence, the constant has to be > lowest value in the data - so we are sure to achieve our goal

100000000>(abs(min(skw_1))+1) # TRUE

setDT(upSampledTrain)
skw_3 <- upSampledTrain[, lapply(.SD,
                                 function(x){
                                   if(is.numeric(x))
                                     return(x+100000000) else return(x)})]


dim(skw_3)

# let us have a look at the distributions of some variables as a check:
histogram(skw_1$V14)
histogram(skw_3$V14)

histogram(skw_1$V111)
histogram(skw_3$V111)

histogram(skw_1$V42)
histogram(skw_3$V42)

hist(skw_1$V30, col="yellow")
hist(skw_3$V30, col="navyblue")


#---- Logarithm transformation

save.image("03-03-2019")

setDF(skw_3)
new_train_x_4_log <- skw_3
new_train_x_4_log[, hig_skw_var] <- 
  sapply(X = skw_3[,hig_skw_var],
         FUN=function(x) x=as.numeric(log(x)))

View(new_train_x_4_log)

# Function to plot the distribution of the skewed variables after the transformation

hig_skw_var

histogram(new_train_x_4_log$V191)
histogram(skw_3$V191)
histogram(skw_1$V191)

histogram(new_train_x_4_log$V150)
histogram(skw_3$V150)
histogram(skw_1$V150)

histogram(new_train_x_4_log$V124)
histogram(skw_3$V124)
histogram(skw_1$V124)

# let s have a look at how thw skewness has changed after the transformation:
skewness(new_train_x_4_log$V124)
skewness(skw_3$V124)
# almost nothing..

# Plot all distributions:
plt_trs_skw(new_train_x_4_log)

#---- Square root transofrmation

new_train_x_4_sqrt <- skw_3
new_train_x_4_sqrt[, hig_skw_var] <- 
  sapply(X = skw_3[,hig_skw_var],
         FUN=function(x) x=as.numeric(sqrt(x)))

View(new_train_x_4_sqrt)

# Plot distributions:
plt_trs_skw(new_train_x_4_sqrt)

hig_skw_var

histogram(new_train_x_4_sqrt$V107)
histogram(skw_3$V107)


#---- Inverse transformation

new_train_x_4_inv <- skw_3
new_train_x_4_inv[, hig_skw_var] <- 
  sapply(X = skw_3[,hig_skw_var],
         FUN=function(x) x=as.numeric((x^-1)))

View(new_train_x_4_inv)

# Plot distributions:
plt_trs_skw(new_train_x_4_inv)

hist(new_train_x_4_inv$V107)
hist(upSampledTrain$V107)


#---- Box-Cox Transformation (UNDER WAY..)

hig_skw_var

require(caret)
test <- skw_3$V1
test <- as.data.frame(as.numeric(test))
names(test)

test <- BoxCoxTrans(test$`as.numeric(test)`)

test$skewness

test$lambda

test$ratio

skewness(skw_3$V1) # mmm..does not change..weird..


#---- Yeo-Jonhson transformation (UNDER WAY)..




###########


# ------------------------ Relief Code



factor_data <- new_train_x_2[,sapply(new_train_x_2, is.factor), with=FALSE]
View(factor_data)
factor_data_1 <- bind_cols(factor_data, train_Y)
factor_data_1$appetency = NULL
factor_data_1$upselling = NULL

sapply(factor_data_1, function(x) length(unique(x)))

reliefValues <- attrEval(churn ~ .,data = factor_data_1, estimator = "ReliefFequalK", ReliefIterations = 50)
sort(reliefValues, decreasing = TRUE)

perm <- permuteRelief(x = factor_data_1[,-1], y = factor_data_n$churn, estimator = "ReliefFequalK", ReliefIterations = 50)
perm_std <- abs(perm$standardized) # since it is t statistic, lets make absolute value

sort(perm_std, decreasing = TRUE)

# with a p value of 0.25 t = 1.96 
# any variable with more than 1.96 should be removed



 
