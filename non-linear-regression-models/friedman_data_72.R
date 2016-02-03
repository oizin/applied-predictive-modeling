# Chapter 7: Non-linear Regression models
#
# Script based on Ex 7.2 of Applied Predictive Modeling (Kuhn & Johnson, 2013)
# Exercise covers on data simulation and model fitting
#
#
# Packages ====================================================================
library(mlbench)
library(caret)
library(reshape2)

# Simulate the data ===========================================================
set.seed(200)
training <- mlbench.friedman1(200, sd = 1)  # data is in list format
training$x <- data.frame(training$x)  # convert predictor matrix to dataframe

# slight pattern in som of the generating variables X1 through X5 compared to the 
# noise variables X6 through X10
featurePlot(training$x, training$y)

# simulate a large test set for greater precision in error estimation
test <- mlbench.friedman1(5000, sd = 1)  
test$x <- data.frame(test$x)

# Fit and test a KNN model ======================================================
ctrl <- trainControl(method = 'repeatedcv', number = 10, repeats = 5)

set.seed(100)
knn_model <- train(x = training$x,
  y = training$y,
  method = "knn",
  preProc = c("center", "scale"),
  trControl = ctrl,
  tuneLength = 10)
knn_model

knn_pred <- predict(knn_model, newdata = test$x)
postResample(pred = knn_pred, obs = test$y)

# the predictions are within a narrower range, tendecy to underpredict high values
# and overpredict low values
plot(x = knn_pred, y = test$y, xlab = 'prediction', ylab = 'observed')
abline(a = 0, b = 1, lty = 2, col = 2)

# Fit and test a SVM model ======================================================
set.seed(100)
svmR_model <- train(x = training$x,
  y = training$y,
  method = "svmRadial",
  preProc = c("center", "scale"),
  trControl = ctrl,
  tuneLength = 10)
svmR_model
svmR_model$finalModel  # epsilon = 0.1

svmR_pred <- predict(svmR_model, newdata = test$x)
postResample(pred = svmR_pred, obs = test$y)

# Much better fit than the knn model
plot(x = svmR_pred, y = test$y, xlab = 'prediction', ylab = 'observed',
  main = 'SVM model')
abline(a = 0, b = 1, lty = 2, col = 2)

# Fit and test a MARS model ====================================================
set.seed(100)
mars_grid <- expand.grid(.degree = 1:2, .nprune = seq(7, 20, by = 2))
mars_model <- train(x = training$x,
  y = training$y,
  method = "earth",
  preProc = c("center", "scale"),
  trControl = ctrl,
  tuneGrid = mars_grid)
mars_model

# Model differentiated signal from noise variables, only X6 was not pruned
# from the models (and then only for degree = 1)
plotmo(mars_model$finalModel)
varImp(mars_model)

mars_pred <- predict(mars_model, test$x)
postResample(pred = mars_pred, obs = test$y)

# Close fit...
plot(x = mars_pred, y = test$y, xlab = 'prediction', ylab = 'observed',
  main = 'MARS model')
abline(a = 0, b = 1, lty = 2, col = 2)

# Fit and test a Neural Networks model =========================================
cor(training$x)

nnet_grid <- expand.grid(.decay = seq(0.5, 1.5, by = 0.1), # book recommends decay is 
  .size = seq(1, 15, by = 2), .bag = FALSE)                # between 0 and 0.1
set.seed(100)
nnet_model <- train(x = training$x,
  y = training$y,
  method = "avNNet",  # avNNet performs model averaging
  preProc = c("center", "scale"),
  trControl = ctrl,
  tuneGrid = nnet_grid,
  linout = TRUE,
  maxit = 500)
nnet_model
nnet_model$finalModel  # there are 5 final models

nnet_pred <- predict(nnet_model, test$x)
postResample(pred = nnet_pred, obs = test$y)

# Slight tendency, as with knn, towards a thinner tailed prediction distribution
# than actually observed
plot(x = nnet_pred, y = test$y, xlab = 'prediction', ylab = 'observed',
  main = 'Neural Networks model')
abline(a = 0, b = 1, lty = 2, col = 2)

nnet_pred_dist <- data.frame(training_data = training$y, predictions = nnet_pred,
  population_data = test$y)
nnet_pred_dist <- melt(nnet_pred_dist); str(nnet_pred_dist)
ggplot(data = nnet_pred_dist, aes(value, ..density.., colour = variable)) + 
  geom_density() +
  theme_

