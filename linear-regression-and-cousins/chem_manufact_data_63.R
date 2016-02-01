# Chapter 6: Linear Regression and its Cousins
#
# Script based on Ex 6.3 of Applied Predictive Modeling (Kuhn & Johnson, 2013)
# Exercise covers on data pre-processing, train/test splitting, missing value 
# imputation, model fitting and assessing variable importance scores
#
# Notes:
#
#
# The data and packages =======================================================
# Response: % yield from a pharmaceutical product's manufacture
# Predictors: biological and manufacturing process variables

library(AppliedPredictiveModeling)
data(ChemicalManufacturingProcess)
str(ChemicalManufacturingProcess)

library(caret)
library(fBasics)
library(corrplot)
library(car)

# Split into training and test data ===========================================
yield <- ChemicalManufacturingProcess$Yield
Split <- createDataPartition(yield, times = 1, p = 0.75)
# predictors...
training <- ChemicalManufacturingProcess[Split$Resample1, -1]
test <- ChemicalManufacturingProcess[-Split$Resample1, -1]
# response ...
resp_train <- yield[Split$Resample1]
resp_test <- yield[-Split$Resample1]

# Preprocess (inlc. missing data) =============================================
preproc_1 <- preProcess(training, 
  method = c("knnImpute", "center", "scale"), k = 5)
# centering and scaling are important when using e.g PLS/PCR as the methods
# seek directions of maximal variation

training <- predict(preproc_1, training)
test <- predict(preproc_1, test)

XX <- cor(train)
corrplot(XX, tl.cex = 0.3)  # some strong correlations
rk(XX)  # rank less than num vars; linear combinations
remove <- findLinearCombos(XX)
training <- training[ ,-remove$remove]
test <- test[ ,-remove$remove]

# what columns should be removed to reduce the correlations?
remove <- findCorrelation(XX, cutoff = .90) 
training_filter <- training[ ,-remove]
test_filter <- test[ ,-remove]

# Transform data - estimate lambda with Yeo Johnson
preproc_3 <- preProcess(training, method = "YeoJohnson")
training <- predict(preproc_3, training)
test <- predict(preproc_3, test)

# distribution of the response
hist(resp_train, xlab = "% yield", main = "Histogram of % yield")

# Model fitting ===============================================================
ctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 5)

# 1. A linear regression model -------------------------------------------------
linear_model <- train(y = resp_train, x = training_filter,
  method = "lm",
  trControl = ctrl)
linear_model
# plots allow search for ill fit or potential quadratic effects
plot(linear_model$finalModel$fitted.values, 
  linear_model$finalModel$residuals, xlab = "fitted", ylab = "residuals")
avPlots(linear_model$finalModel)

# how does the final model fit look vs. the data?
plot(linear_model$finalModel$fitted.values, resp_train, xlab = "observed",
  ylab = "predicted")
abline(a = 0, b = 1, col = 2)
cor(linear_model$finalModel$fitted.values, resp_train)^2  # R2 fit

# 2. Principal components regression ------------------------------------------
pcr_results <- list(results = data.frame(RMSE = NA, RMSE_sd = NA), final = NA)
for (i in 1:20) {
  # fit model
  train_data <- princomp(training)$scores[ ,1:i]
  train_data <- data.frame(train_data)
  pcr_model <- train(y = resp_train,
    x = train_data,
    method = "lm",
    trControl = ctrl)
  
  # extract results
  pcr_results$results[i, 1] <- pcr_model$results$RMSE
  pcr_results$results[i, 2] <- pcr_model$results$RMSESD
  
  # extract model
  if (all(pcr_model$results$RMSE <= pcr_results$results$RMSE)) {
    pcr_results$final <- pcr_model
  }
}
pcr_results
xyplot(pcr_results$results$RMSE ~ 1:20, xlab = "ncomp", ylab = "RMSE")

# 3. Partial least squares -----------------------------------------------------
pls_model <- train(y = resp_train, x = training,
  method = "pls",
  trControl = ctrl,
  tuneLength = 10)
pls_model
plot(pls_model)

# 4. Ridge regression ----------------------------------------------------------
ridge_grid <- expand.grid(.lambda = seq(0.05, 0.2, 0.01))
ridge_model <- train(y = resp_train, x = training,
  method = "ridge",
  trControl = ctrl,
  tuneGrid = ridge_grid)
ridge_model
plot(ridge_model)

# y-axis: coefficient, x-axis: shrinkage proportion
plot(ridge_model$finalModel)

# 5. LASSO ----------------------------------------------------------------------
lasso_grid <- expand.grid(.fraction = seq(0.01, 0.20, 0.01))
lasso_model <- train(y = resp_train, x = training,
  method = "lasso",
  trControl = ctrl,
  tuneGrid = lasso_grid)
lasso_model
plot(lasso_model)

# 6. Elastic Net ---------------------------------------------------------------
enet_model <- train(y = resp_train, x = training,
  method = "enet",
  trControl = ctrl,
  tuneLength = 10)
enet_model
plot(enet_model)

# Make predictions ============================================================
# 1. linear regression model
lm_preds <- predict(linear_model, test_filter)
RMSE(lm_preds, resp_test)
plot(lm_preds, resp_test, xlab = "prediction", ylab = "observed", pch = 20,
  main = "Linear Regression: RMSE = 1.172")
abline(a = 0, b = 1, col = 2, lty = 2)

# 2. PCR model 
pca_object <- princomp(training)
test_pc <- predict(pca_object, test)
pcr_preds <- predict(pcr_results$final, test_pc)
RMSE(pcr_preds, resp_test)
plot(pcr_preds, resp_test, xlab = "prediction", ylab = "observed", pch = 20,
  main = "PCR: RMSE = 1.144")
abline(a = 0, b = 1, col = 2, lty = 2)

# 3. PLS model 
pls_preds <- predict(pls_model, test)
RMSE(pls_preds, resp_test, na.rm = TRUE)
plot(pls_preds, resp_test, xlab = "prediction", ylab = "observed", 
  pch = 20, main = "Partial Least Squares: RMSE = 1.079")
abline(a = 0, b = 1, col = 2, lty = 2)

# 4. Ridge model
ridge_preds <- predict(ridge_model, test)
RMSE(ridge_preds, resp_test)
plot(ridge_preds, resp_test, xlab = "prediction", ylab = "observed", 
  pch = 20, main = "Ridge Reg: RMSE = 1.086")
abline(a = 0, b = 1, col = 2, lty = 2)

# 4. LASSO model
lasso_preds <- predict(lasso_model, test)
RMSE(lasso_preds, resp_test)
plot(lasso_preds, resp_test, xlab = "prediction", ylab = "observed", 
  pch = 20, main = "LASSO: RMSE = 1.112")
abline(a = 0, b = 1, col = 2, lty = 2)

# 5. enet model
enet_preds <- predict(enet_model, test)
RMSE(enet_preds, resp_test)
plot(enet_preds, resp_test, xlab = "prediction", ylab = "observed", pch = 20,
  main = "Elastic Net: RMSE = 1.040")
abline(a = 0, b = 1, col = 2, lty = 2)

# Variable Importance =========================================================
varImp(pls_model)
varImp(ridge_model)
varImp(enet_model)

plot(training$ManufacturingProcess13, resp_train, pch = 20, 
  xlab = "ManufacturingProcess13", ylab = "% yield")
lines(loess.smooth(training$ManufacturingProcess13, resp_train), col = 2)

plot(training$ManufacturingProcess32, resp_train, pch = 20, 
  xlab = "ManufacturingProcess32", ylab = "% yield")
lines(loess.smooth(training$ManufacturingProcess32, resp_train), col = 2)

plot(training$ManufacturingProcess36, resp_train, pch = 20, 
  xlab = "ManufacturingProcess36", ylab = "% yield")
lines(loess.smooth(training$ManufacturingProcess36, resp_train), col = 2)

plot(training$ManufacturingProcess36, resp_train, pch = 20, 
  xlab = "ManufacturingProcess36", ylab = "% yield")
lines(loess.smooth(training$ManufacturingProcess36, resp_train), col = 2)

# The variable importance scores reveal several relationships worth examining in 
# any experiments aimed at improving the manufacturing process.