# Exercise 3.1
# Visual data exploration and possible transformations

library(mlbench)
data(Glass)  # 6 types of glass; defined in terms of their oxide content (i.e. Na, Fe, K, etc)
str(Glass)

# Correlation matrix of the variables
library(corrplot)
Glass.corr <- cor(Glass[ ,1:9])  # correlation matrix of predictors
corrplot(Glass.corr, method = "number", type = "upper")  # correlation matrix visualisation
plot(Glass$RI, Glass$Ca, main="Scatterplot of RI and Ca") # evidence of linear relationship

# Histogram of the variables
library(reshape2)
library(ggplot2)
Glass.stack <- melt(Glass)  # 3 columns w/ Type and variable name as factor variables
str(Glass.stack)
p <- ggplot(Glass.stack, aes(x = value))
p + geom_histogram(col="blue") + facet_wrap(~ variable, scales = "free")
# RI, Na, Si, AL, Ca are relatively normal though with outliers
# Ba, Fe, K and Mg have large proportions of zero/near zero values
table(Glass$Ba==0)/length(Glass$Ba)

library(caret)
nearZeroVar(Glass, saveMetrics = TRUE)  # Fe, Ba, K have large numbers of zero values

# Skewness and Box Cox lamba estimates
library(e1071)
skewValues <- apply(Glass.pred, 2, skewness)
skewValues  # K > Ba > Ca > Fe > RI (all over |1|)
BoxCoxTrans(Glass.pred$K)  # could not estimate
BoxCoxTrans(Glass.pred$Ba)  # could not estimate
BoxCoxTrans(Glass.pred$Ca)  # -1
BoxCoxTrans(Glass.pred$Fe)  # could not estimate
BoxCoxTrans(Glass.pred$RI)  # -2

# Principal Component Analysis
pcaObject <- prcomp(Glass.pred, center = TRUE, scale. = TRUE)
percentVariance <- pcaObject$sd^2/sum(pcaObject$sd^2)*100  # variance each component a/c for
round(percentVariance, 2)
round(head(pcaObject$rotation[ ,1:5]),2)  # variable loadings



