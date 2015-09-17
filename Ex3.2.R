# Ex3.2
# Exploratory analysis focusing on identifying missing values and near zero variance

library(mlbench)
data(Soybean) 
?Soybean  # dna = does not apply
str(Soybean)  # factors and ordinal 

# Near Zero Variance?
library(caret)
library(ggplot2)
SoyZeroVar <- nearZeroVar(Soybean, freqCut = 95/5)  # 3 variables
qplot(Soybean[ ,19])
qplot(Soybean[ ,26])
qplot(Soybean[ ,28])
nearZeroVar(Soybean, freqCut = 90/10)  # 8 variables

# Missing Values a) predictors
table(is.na(Soybean))/(683*36)
tab.na <- matrix(data = NA, nrow = 2, ncol = 35)
rownames(tab.na) <- c("NA", "not NA")
colnames(tab.na) <- colnames(Soybean[-1])  # [-1] to leave out Class
for (i in 2:36) { # create a table with NA vs not NA values for each variable
  tab.na[ , i-1] <- table(is.na(Soybean[ , i]))/(length(Soybean[ , i]))
}
tab.na <- tab.na[ , order(tab.na[1, ])]
barplot(tab.na[1, -12], cex.names = 0.8, las = 2, main = "Proportion of non-missing values")
library(mi)  # an approach using the pkg mi
mSoybean <- missing_data.frame(Soybean)
image(mSoybean)

# Missing Values b) response
table(is.na(Soybean[,1]))  # all the class obs are present
for (i in 1:683) {  # loop to count missing values per row
  Soybean[i, 37] <- sum(is.na(Soybean)[i, ])
}
colnames(Soybean)[37] <- "missing"
Soybean.na <- subset(Soybean, select = c(Class, missing))  # subset by Class
Soybean[ , 37] <- NULL  # rm extra column
missing.by.Class <- aggregate(missing~Class, data = Soybean.na, FUN = "sum")
plot(missing.by.Class, cex.axis = 0.7, las = 2, xlab = "", main = "Missing Values per Class")
# phytophthora-rot most impacted by missing values, followed by 2-4-d-injury and cyst-nematode

# Dealing with the missing values / near zero var
library(ipred)
Soybean.filtered <- Soybean[ , -SoyZeroVar] # rm zero var variables
dummies <- dummyVars(Class ~ ., data = Soybean.filtered, na.action = na.pass)
Soybean.filtered <- (predict(dummies, newdata = Soybean))  # dummy variables
preProcess(Soybean.filtered, method="bagImpute")  # impute missing values using bagged tree method

rm(ls = list())
