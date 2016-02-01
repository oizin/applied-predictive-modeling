# Chapter 7: Non-linear Regression models
#
# Script based on Ex 7.1 of Applied Predictive Modeling (Kuhn & Johnson, 2013)
# Exercise covers on data simulation, train/test splitting, the SVM tuning parameters
#
## SVM Notes:
# Can be seen as a form of robust regression.
# e-insensitive regression: data points with residuals inside the e boundary do not 
# contribute to the model while those outside contribute a linear scaled amount.
# MIN Cost*SUM[L(y - y_hat)] + SUM(beta^2); L is the e-insensitve function.
# Over-parameterised model, as many beta as points, however these are zero for 
# points inside the e boundary.
# The support vectors are those points with non-zero beta (or alpha!).
# New points enter as dot-product of the support vectors. An extension to non-linear 
# situations is to use other Kernel function around the dot product (e.g. radial
# basis). Non-linear kernels have scaling parameters requiring tuning.
#
#
# Packages ====================================================================
library(kernlab)

# Simulate the data ===========================================================
set.seed(100)
x <- runif(100, min = 2, max = 10)
y <- sin(x) + rnorm(length(x)) * .25
sin_data <- data.frame(x = x, y = y)
plot(x, y)

# Create a grid of x values to use for prediction
data_grid <- data.frame(x = seq(2, 10, length = 100))

# Fit different models using a radial basis function and different values of ===
# the cost (the C parameter).
radial_SVM.1 <- ksvm(x = x, y = y, data = sin_data,
  kernel ="rbfdot", kpar = "automatic",
  C = 1, epsilon = 0.01)
model_preds.1 <- predict(radial_SVM.1, newdata = data_grid)
# very jagged line, over-fits the data
points(x = data_grid$x, y = model_preds.1[,1], type = "l", col = "blue")

radial_SVM.2 <- ksvm(x = x, y = y, data = sin_data,
  kernel ="rbfdot", kpar = "automatic",
  C = 1, epsilon = 0.1)
model_preds.2 <- predict(radial_SVM.2, newdata = data_grid)
# somewhat smoother fit but still not exactly sin wave
points(x = data_grid$x, y = model_preds.2[,1], type = "l", col = "red")

radial_SVM.3 <- ksvm(x = x, y = y, data = sin_data,
  kernel ="rbfdot", kpar = "automatic",
  C = 2, epsilon = 0.3)
model_preds.3 <- predict(radial_SVM.3, newdata = data_grid)
# nearly perfect, increased cost and epsilon vs. others...
points(x = data_grid$x, y = model_preds.3[,1], type = "l", col = "green")

radial_SVM.4 <- ksvm(x = x, y = y, data = sin_data,
  kernel ="rbfdot", kpar = "automatic",
  C = 10, epsilon = 0.01)  # high cost, low e
model_preds.4 <- predict(radial_SVM.4, newdata = data_grid)
# with very high cost starts to be influenced by outliers
points(x = data_grid$x, y = model_preds.4[,1], type = "l", col = "orange")

radial_SVM.5 <- ksvm(x = x, y = y, data = sin_data,
  kernel ="rbfdot", kpar = "automatic",
  C = 1, epsilon = 0.05)  # high cost, low e
model_preds.5 <- predict(radial_SVM.5, newdata = data_grid)
# matches the green almost exactly
points(x = data_grid$x, y = model_preds.5[,1], type = "l", col = "purple")

# real function
points(sort(x), sin(sort(x)), type = "l")

# Changing sigma ------------------------------------------------------------
plot(x, y)

radial_SVM.1 <- ksvm(x = x, y = y, data = sin_data,
  kernel ="rbfdot", kpar = list(sigma = 1),
  C = 2, epsilon = 0.1)  # high cost, low e
model_preds.1 <- predict(radial_SVM.1, newdata = data_grid)
points(x = data_grid$x, y = model_preds.1[,1], type = "l", col = "purple")

radial_SVM.2 <- ksvm(x = x, y = y, data = sin_data,
  kernel ="rbfdot", kpar = list(sigma = 3),
  C = 2, epsilon = 0.1)  # high cost, low e
model_preds.2 <- predict(radial_SVM.2, newdata = data_grid)
points(x = data_grid$x, y = model_preds.2[,1], type = "l", col = "green")

radial_SVM.3 <- ksvm(x = x, y = y, data = sin_data,
  kernel ="rbfdot", kpar = list(sigma = 5),
  C = 2, epsilon = 0.1)  # high cost, low e
model_preds.3 <- predict(radial_SVM.3, newdata = data_grid)
points(x = data_grid$x, y = model_preds.3[,1], type = "l", col = "orange")

radial_SVM.4 <- ksvm(x = x, y = y, data = sin_data,
  kernel ="rbfdot", kpar = list(sigma = 7),
  C = 2, epsilon = 0.1)  # high cost, low e
model_preds.4 <- predict(radial_SVM.4, newdata = data_grid)
points(x = data_grid$x, y = model_preds.4[,1], type = "l", col = "blue")

radial_SVM.5 <- ksvm(x = x, y = y, data = sin_data,
  kernel ="rbfdot", kpar = list(sigma = 0.1),
  C = 2, epsilon = 0.1)  # high cost, low e
model_preds.5 <- predict(radial_SVM.5, newdata = data_grid)
points(x = data_grid$x, y = model_preds.5[,1], type = "l", col = "red")

radial_SVM.6 <- ksvm(x = x, y = y, data = sin_data,
  kernel ="rbfdot", kpar = list(sigma = 0),
  C = 2, epsilon = 0.1)  # high cost, low e
model_preds.6 <- predict(radial_SVM.6, newdata = data_grid)
points(x = data_grid$x, y = model_preds.6[,1], type = "l", col = "yellow")

points(x = data_grid$x, y = rep(mean(y), length(x)), type = "l")

# sigma determines the extent smoothing, if zero left only with beta_0...
# Cost determines error impact on model, e determines number of support vectors
# book suggests fixing e and tuning the cost (there is a relationship between the two)
