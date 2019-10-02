##### BEST PERFORMANCE MODEL FOR UPSELLING 


# loading packages

setwd("~/Desktop/Upselling_Preprocessing")

load("model2_GBM_IMA") # this contains previous steps and variables we'll need in this model

library(caret)
library(DMwR)
library(parallel)
library(doParallel)


# SMOTE transformation and SPatial Sign transformation
rf_smote_train <- SMOTE(upselling~., data=upselling_train_centered) ### this dataset is a center and scaled version of dataset
summary(rf_smote_train$upselling)

names(rf_smote_train)

rf_smote_train_pre <- preProcess(rf_smote_train[,-22],
                                 method=c("spatialSign"))

rf_smote_train_pre <- predict(rf_smote_train_pre,rf_smote_train)

fiveStats <- function(...) c(twoClassSummary(...),
                             defaultSummary(...))


#####################  model 1 #######


# trainControl (1o-fold cross validation repeated 1 time)
rf_control <- trainControl(method="repeatedcv",
                           repeats=1,
                           classProbs = TRUE,
                           summaryFunction = fiveStats,
                           search="grid",
                           allowParallel = TRUE)



# The grid:
rf_tunegrid_2 <- expand.grid(.mtry=c(round(sqrt(ncol(rf_smote_train_pre)-1)),2,5))

# The model:
library(doParallel)
cl <- makeCluster(detectCores())
registerDoParallel(cl)

set.seed(123)
rf_gridsearch_2 <- train(make.names(upselling)~.,
                         data= rf_smote_train_pre,
                         method="rf",
                         metric="ROC",
                         tuneGrid=rf_tunegrid_2,
                         trControl=rf_control)


print(rf_gridsearch_2)



########## AUC UNDER TRAPEZOID #############


TPR_2 <- rf_gridsearch_2$results$Sens # Sensitivity
FPR_2 <- (1-(rf_gridsearch_2$results$Spec)) # 1-Specificity
TPZ_AUC_2 <- (0.5*(TPR_2*FPR_2)) + (0.5*((TPR_2)+1)*(1-FPR_2))  # Area of Trangle + Area of Trapezoid = AUC

print(TPZ_AUC_2) ### best result is 0.9475356



############   closing and saving



model_2_rf_ups <- capture.output(rf_gridsearch_2)
cat(model_2_rf_ups, file="model2_RandomForest_IMA.txt", sep="\n")


save.image("best_RandomForest_IMA")

load("best_RandomForest_IMA")



############## predictions ##############

best_upselling <- predict(rf_gridsearch_2,upselling_train_centered_test)
best_upselling_2 <- as.data.frame(best_upselling)

table(best_upselling_2)

levels(best_upselling)[levels(best_upselling)=="X.1"] <- "1"
levels(best_upselling)[levels(best_upselling)=="X1"] <- "-1"
best_upselling_4 <- data.frame(best_upselling)

names(best_upselling_4) <- "upselling"

table(best_upselling_4)

## 1    - 1 
## 793 10934

write.csv(best_upselling_4,"ups_bst_predictions.csv")


