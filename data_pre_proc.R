# Chapter 3: Data Pre-Processing
#
# Script based on Ex 3.1 to 3.3 of Applied Predictive Modeling (Kuhn & Johnson, 2013)
# Exercises focus on data pre-processing incl. missingness, distributions
# (skewness/outliers), transformations, near zero variance and visualising above...
#
# ==============================================================================
# Exercise 3.1 
# AIM: Visual data exploration (distributions/outliers) and explore possible
# transformations
#
# Notes:
#
#
# The data ---------------------------------------------------------------------
# 6 types of glass; defined in terms of their oxide content (i.e. Na, Fe, K, etc)
library(mlbench)
data(Glass)
str(Glass)

library(corrplot)
library(reshape2)
library(ggplot2)
library(caret)
library(e1071)

# Correlation matrix of the variables ------------------------------------------
Glass.corr <- cor(Glass[ ,1:9])  # correlation matrix of predictors
corrplot(Glass.corr, method = "number", type = "upper")  # correlation matrix visualisation
plot(Glass$RI, Glass$Ca, 
  main = "Scatterplot of RI and Ca") # evidence of linear relationship

# Histogram of the variables ---------------------------------------------------
Glass.stack <- melt(Glass)  # 3 columns w/ Type and variable name as factor variables
str(Glass.stack)
p <- ggplot(Glass.stack, aes(x = value))
p + geom_histogram(col = "blue") + facet_wrap(~ variable, scales = "free")
# RI, Na, Si, AL, Ca are relatively normal though with outliers
# Ba, Fe, K and Mg have large proportions of zero/near zero values
table(Glass$Ba == 0)/length(Glass$Ba)

# Near zero variances ----------------------------------------------------------
nearZeroVar(Glass, saveMetrics = TRUE)  # saveMetrics returns useful info
# Fe, Ba, K have large numbers of zero values

# Transformations: Skewness and Box Cox lamba estimates -----------------------
Glass.pred <- Glass[ ,1:9]
skewValues <- apply(Glass.pred, 2, skewness)
skewValues  # K > Ba > Ca > Fe > RI (all over |1|)
apply(Glass.pred[abs(skewValues) > 1], 2, BoxCoxTrans)  # BoxCox estimates

# Principal Component Analysis ------------------------------------------------
pcaObject <- prcomp(Glass.pred, center = TRUE, scale. = TRUE)
percentVariance <- pcaObject$sd^2/sum(pcaObject$sd^2)*100  # variance each component a/c for
round(percentVariance, 2)
round(head(pcaObject$rotation[ ,1:5]),2)  # variable loadings

rm(list = ls())  # clear enviroment

# ==============================================================================
# Exercise 3.2 
# AIM: Exploratory analysis focusing on identifying missing values and 
# near zero variance
#
# Notes:
#
#
# The data ---------------------------------------------------------------------
library(mlbench)
data(Soybean) 
?mlbench::Soybean  # dna = does not apply
str(Soybean)  # factors and ordinal factors

library(caret)
library(ggplot2)
library(mi)  # for visualising missing values
library(ipred)  # imputation

# Near Zero Variances ---------------------------------------------------------
SoyZeroVar <- nearZeroVar(Soybean, freqCut = 95/5); SoyZeroVar  # 3 variables
qplot(Soybean[ ,19])
qplot(Soybean[ ,26])
qplot(Soybean[ ,28])
nearZeroVar(Soybean, freqCut = 90/10)  # 8 variables

# Missing Values a) Predictors -----------------------------------------------
table(is.na(Soybean))/(683*36)  # how many missing values?
tab.na <- matrix(data = NA, nrow = 2, ncol = 35)  # init matrix for storing NA info
rownames(tab.na) <- c("NA", "not NA")
colnames(tab.na) <- colnames(Soybean[-1])  # [-1] to leave out Class

# Create a table with NA vs not NA values for each variable
for (i in 2:36) { 
  tab.na[ , i-1] <- table(is.na(Soybean[ , i]))/(length(Soybean[ , i]))
}
tab.na <- tab.na[ ,order(tab.na[1, ])]

# Visualise the results
barplot(tab.na[1, -12], cex.names = 0.8, las = 2, 
  main = "Proportion of non-missing values")

# Alternate visualisation
mSoybean <- missing_data.frame(Soybean)
image(mSoybean)

# Missing Values b) response (incl. general data missingness by response level)
table(is.na(Soybean[,1]))  # all the class obs are present

for (i in 1:683) {  # loop to count missing values per row
  Soybean[i, 37] <- sum(is.na(Soybean)[i, ])
}
colnames(Soybean)[37] <- "missing"
Soybean.na <- subset(Soybean, select = c(Class, missing))  # subset by Class
Soybean[ , 37] <- NULL  # rm extra column
missing.by.Class <- aggregate(missing ~ Class, data = Soybean.na, FUN = "sum")
plot(missing.by.Class, 
  cex.axis = 0.7, 
  las = 2, 
  xlab = "", 
  main = "Missing Values per Class")
# phytophthora-rot most impacted by missing values, followed by 2-4-d-injury and cyst-nematode

# Initial attempt to deal with the missing values / NZV -----------------------
Soybean.filtered <- Soybean[ , -SoyZeroVar] # rm zero var variables
dummies <- dummyVars(Class ~ ., data = Soybean.filtered, na.action = na.pass)
Soybean.filtered <- predict(dummies, newdata = Soybean)  # dummy variables
preProcess(Soybean.filtered, method="bagImpute")  # impute missing values using 
# bagged tree method

rm(list = ls())  # clear enviroment

# ==============================================================================
# Exercise 3.3
# AIM: Data exploration with a focus on associated predictors / near zero variance
#
# Notes:
#
#
# The data ---------------------------------------------------------------------
library(caret)
data(BloodBrain)
?BloodBrain  # each of the 208 compounds have multiple descriptors
str(bbbDescr)

library(corrplot); ?corrplot
library(e1071)

# Correlation matrix of the variables ------------------------------------------
bbbDescr.corr <- cor(bbbDescr)
corrplot(bbbDescr.corr, method = "shade", tl.cex = 0.4, order = "hclust")
# corrplot indicates evidence of linear relationship between many variables

highCorr <- findCorrelation(bbbDescr.corr, cutoff = .75)  # which variables?
length(highCorr)  # 66 variables included
bbbDescr.filtered <- bbbDescr[, -highCorr]  # rm algorithm selected high correlations

# Skewness and transformations ------------------------------------------------
SkewValues <- apply(bbbDescr, 2, skewness)
SkewedVariables <- subset(SkewValues, abs(SkewValues) > 1)
length(SkewedVariables)  # 71 variables

# BoxCox Transformations
preProc <- preProcess(bbbDescr.filtered, method = "BoxCox")
bbbDescr.proc <- predict(preProc, bbbDescr.filtered)

# Near Zero Variance ----------------------------------------------------------
nearZeroVar(bbbDescr)  # 7 variabless
bbbDescr.nearzero <- nearZeroVar(bbbDescr.proc) 
bbbDescr.final <- bbbDescr.proc[ , -bbbDescr.nearzero]

# Leading to the final dataframe -----------------------------------------------
# remaining correlations
corrplot(cor(bbbDescr.final), method = "shade", tl.cex = 0.4, order = "hclust")
# skewness following tranformations
table(apply(bbbDescr.final, 2, skewness) > 1)  

rm(list = ls())  # clear enviroment



