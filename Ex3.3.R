# Ex3.3
# Data exploration with a focus on associated predictors / near zero variance

library(caret)
data(BloodBrain)
?BloodBrain  # each of the 208 compounds have multiple descriptors
str(bbbDescr)

# Correlation matrix of the variables
library(corrplot); ?corrplot
library(caret )
bbbDescr.corr <- cor(bbbDescr)
corrplot(bbbDescr.corr, method = "shade", tl.cex = 0.4, order = "hclust")
# corrplot indicates evidence of linear relationship between many variables
highCorr <- findCorrelation(bbbDescr.corr, cutoff = .75)
length(highCorr)  # 66 variables included
bbbDescr.filtered <- bbbDescr[, -highCorr]  # rm algorithm selected high correlations

# Skewness
library(e1071)
SkewValues <- apply(bbbDescr, 2, skewness)
SkewedVariables <- subset(SkewValues, abs(SkewValues) > 1)
length(SkewedVariables)  # 71 variables

# BoxCox Transformations
preProc <- preProcess(bbbDescr.filtered, method = "BoxCox")
bbbDescr.proc <- predict(preProc, bbbDescr.filtered)

# Near Zero Variance
nearZeroVar(bbbDescr)  # 7 variable
bbbDescr.nearzero <- nearZeroVar(bbbDescr.proc) 
bbbDescr.final <- bbbDescr.proc[ , -bbbDescr.nearzero]

# Final dataframe
corrplot(cor(bbbDescr.final), method = "shade", tl.cex = 0.4, order = "hclust")  
# far less variables after removing highly correlated variables
table(apply(bbbDescr.final, 2, skewness) > 1)  # still a number of skewed variables




