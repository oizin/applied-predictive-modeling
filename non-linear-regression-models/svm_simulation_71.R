# Chapter 7: Non-linear Regression models
#
# Script based on Ex 7.1 of Applied Predictive Modeling (Kuhn & Johnson, 2013)
# Exercise covers on data simulation and the SVM tuning parameters
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
library(RColorBrewer)

# Simulate the data ===========================================================
set.seed(100)
x <- runif(100, min = 2, max = 10)
y <- sin(x) + rnorm(length(x)) * .25
sin_data <- data.frame(x = x, y = y)
plot(x, y)

# Create a grid of x values to use for prediction
data_grid <- data.frame(x = seq(2, 10, length = 100))

# Fit different models using a radial basis function and different values of ===
# the cost (the C parameter)

mypalette <- brewer.pal(9, "Greens")

par(mar=c(5.1, 4.1, 4.1, 8.1), xpd=TRUE)
plot(x, y)
for (i in 1:9) {
  radial_SVM <- ksvm(x = x, y = y, data = sin_data,
    kernel ="rbfdot", kpar = "automatic",
    C = i/2, epsilon = 0.1)  # move the cost from 0.5 to 4.5
  model_preds <- predict(radial_SVM, newdata = data_grid)
  points(x = data_grid$x, y = model_preds[,1], type = "l", col = mypalette[i])
}
legend("topright", inset=c(-0.2,0), lty = 1, legend=c(seq(0.5, 4.5, by = 0.5)), 
  col = mypalette[1:10], title="Cost", cex = 0.5)
par(xpd = FALSE)
points(seq(1, 11, by = 0.1), sin(seq(1, 11, by = 0.1)), type = "l", lwd = 2)

# Fit different models using a radial basis function and different values of ===
# the epsilon boundary parameter

mypalette <- brewer.pal(9, "Blues")

par(mar=c(5.1, 4.1, 4.1, 8.1), xpd=TRUE)
plot(x, y, bty='L')
for (i in 1:9) {
  radial_SVM <- ksvm(x = x, y = y, data = sin_data,
    kernel ="rbfdot", kpar = "automatic",
    C = 1, epsilon = i/10) # move epsilon from 0.1 to 0.9
  model_preds <- predict(radial_SVM, newdata = data_grid)
  points(x = data_grid$x, y = model_preds[,1], type = "l", 
    col = mypalette[i], 
    lwd = 2)
}
legend("topright", inset=c(-0.2,0), lty = 1, legend=c(seq(0.1, 0.9, by = 0.1)), 
  col = mypalette[1:10], title="Epsilon", cex = 0.5)
par(xpd = FALSE)
points(seq(1, 11, by = 0.1), sin(seq(1, 11, by = 0.1)), type = "l", lwd = 1)

# Fit different models using a radial basis function and different values of ===
# sigma (the scaling parameter)

mypalette <- brewer.pal(9, "Reds")

par(mar=c(5.1, 4.1, 4.1, 8.1), xpd=TRUE)
plot(x, y)
for (i in 1:9) {
  radial_SVM <- ksvm(x = x, y = y, data = sin_data,
    kernel ="rbfdot", kpar = list(sigma = i), # move epsilon from 1 to 9
    C = 1, epsilon = 0.1) 
  model_preds <- predict(radial_SVM, newdata = data_grid)
  points(x = data_grid$x, y = model_preds[,1], type = "l", 
    col = mypalette[i], 
    lwd = 2)
}
legend("topright", inset=c(-0.2,0), lty = 1, legend=c(seq(1, 9, by = 1)), 
  col = mypalette[1:10], title="Sigma", cex = 0.5)
par(xpd = FALSE)
points(seq(1, 11, by = 0.1), sin(seq(1, 11, by = 0.1)), type = "l", lwd = 1)


# sigma determines the extent smoothing, if zero left only with beta_0...
# Cost determines error impact on model, e determines number of support vectors
# book suggests fixing e and tuning the cost (there is a relationship between the two)
