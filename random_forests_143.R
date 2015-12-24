# Classification Trees and Rule Based Models
#
# Script based on Ex 14.3 of Applied Predictive Modeling (Kuhn & Johnson, 2013)
#
# AIM: investigate difference between CART and conditional inference random
# forest models
#
# In particular:
# a) The resulting models - 10-fold cv performance, optimal mtry
# b) Fitting time
# c) Most important variables - any differences?
# d) Impact of preprocessing on CART most important variables
#
# Load data and packages =======================================================
library(caret)
library(AppliedPredictiveModeling)
library(party)

data(hepatic)


# Fit a random forest model using both CART trees and conditional inference trees
set.seed(714)
indx <- createFolds(injury, returnTrain = TRUE)
ctrl <- trainControl(method = "cv", index = indx)
mtryValues <- c(5, 10, 25, 50, 75, 100)

# CART random forest
rfCART <- train(chem, injury,  
  method = "rf",
  metric = "Kappa",
  ntree = 1000,
  tuneGrid = data.frame(.mtry = mtryValues), 
  do.trace = TRUE)
rfCART

# Conditional inference random forest
rfcForest <- train(chem, injury,  # conditional inference trees
  method = "cforest",
  metric = "Kappa",
  tuneGrid = data.frame(.mtry = mtryValues))
rfcForest


# Fitting times ---------------------------------------------------------------
rfCART$times$everything
rfcForest$times$everything


# Variable importance ---------------------------------------------------------
varImp(rfCART)
temp <- varImp(rfCART)$importance
imp_var_rfCART <- rownames(temp)[order(temp$Overall, decreasing = TRUE)][1:20]
round(sapply(chem[imp_var_rfCART], var), 2)  # variance of variables
sapply(chem[imp_var_rfCART], range)  # range of variables


varImp(rfcForest)
temp <- varImp(rfcForest)$importance
imp_var_rfcForest <- rownames(temp)[order(temp$Overall, decreasing = TRUE)][1:20]
round(sapply(chem[imp_var_rfcForest], var), 2)  # variance of variables
sapply(chem[imp_var_rfcForest], range)  # range of variables


# The preference of the CART model for continuous rather than binary or discrete
# variables that take on few values is clear from the variances of the top 20 
# variables of each model. 
#
# Does preprocessing CART models reduce this bias? ----------------------------

rfCART.2 <- train(chem, injury,  
  method = "rf",
  metric = "Kappa",
  ntree = 1000,
  tuneGrid = data.frame(.mtry = mtryValues), 
  do.trace = TRUE,
  preProcess = c("center", "scale"))
rfCART.2

varImp(rfCART.2)
temp <- varImp(rfCART.2)$importance
imp_var_rfCART.2 <- rownames(temp)[order(temp$Overall, decreasing = TRUE)][1:20]
round(sapply(chem[imp_var_rfCART.2], var), 2)  # variance of variables
sapply(chem[imp_var_rfCART.2], range)  # range of variables

# Pre-processing alters the variables considered most important although cross
# validation performance is reduced. The preference towards high variance
# variables taking on many values is reduced. A different mtry value is return.


# Random Forests (CART vs cforest) ============================================
# CART random forest: Trees are high variance, low bias. Bagging attempts to 
# reduce the variance by bootstrap aggregating a number of trees. However as 
# each tree uses the same set of predictors they are not independent of one 
# another. Random forest attempts to deal with this by selecting a subset m of 
# the full set predictors at each split. 
# For categorical response the m is typically set at sqrt(p), p being the full
# set of predictors. The trees are then fit as normal, i.e. using the Gini index
# or an information criterion for classification.
#
# One issue with CART random forest is a bias in variable selection, specifically
# a preference for continuous over categorical variables. Conditional inference 
# trees (Hothorn et al. 2006) are one technique to overcome this. These trees 
# are fit by performing a hypothesis test at each candidate split (across an 
# exhaustive list) and generating p-values. This allows p-values rather than 
# raw differences to be compared which decreases bias as they are on the same 
# scale. Multiple comparison correction is also used, as such trees are not pruned 
# as increased splits result in reduced power of the test and less likelihood
# of false positive splits
