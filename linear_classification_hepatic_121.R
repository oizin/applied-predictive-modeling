# Linear Classification Methods
#
# Script based on Ex 12.1 of Applied Predictive Modeling (Kuhn & Johnson, 2013)
#
# AIM: predict whether a compound is likely to cause liver damage based 
# on biological and chemical variables
#
# In particular answering:
# a) What is a good train/test split given an imbalanced response
# b) Which of the linear classification techniques works best and what are the
# most important variables
# c) How does performance compare based on the bio and chem sata sets separately
# and together
#
# Load data and packages ======================================================

library(caret)
library(AppliedPredictiveModeling)
data(hepatic)
?AppliedPredictiveModeling::bio

library(MASS)
library(corrplot)

# Create a training and test set ==============================================
table(injury)  # table of response levels
length(injury)  # how many obs

# Stratified sample: i.e. sample by factor level
# Create stratified sampling function
stratified_sample <- function(x, p) {  
  
  out <- NULL  # initialise output
  
  for (i in 1:length(levels(x))) {
  
    strata <- levels(x)[i]  # factor level to sample
    
    temp <- sample(which(x %in% strata),  # take sample by current level
      size = length(x[x == strata])*p, 
      replace = FALSE)
    
    out <- c(out, temp)  # add sample to previous sample
    
  }
  out <- sort(out)
  out  # output vector
}

training_index <- stratified_sample(x = injury, p = 0.6)

# Create the training/test sets
training_bio <- bio[training_index, ]
test_bio <- bio[-training_index, ]
training_chem <- chem[training_index, ]
test_chem <- chem[-training_index, ]

rm(bio, chem)  # remove unnecessary data


# Pre-process the data ========================================================
# 1. Near Zero Variance
nearZeroVar(training_bio, freqCut = 99/1, saveMetrics = TRUE)
remove <- nearZeroVar(training_bio, freqCut = 95/5)
training_bio <- training_bio[ ,-remove]

nearZeroVar(training_chem, freqCut = 99/1, saveMetrics = TRUE)
remove <- nearZeroVar(training_chem, freqCut = 95/5)
training_chem <- training_chem[ ,-remove]

# 2. Correlations
corr_bio <- cor(training_bio)
corrplot(corr_bio, tl.cex = 0.5)
remove <- findCorrelation(corr_bio)
training_bio <- training_bio[ ,-remove]

corr_chem <- cor(training_chem)
corrplot(corr_chem, tl.cex = 0.5)
remove <- findCorrelation(corr_chem)
training_chem <- training_chem[ ,-remove]

# 3. Transform, center and scale
preProc_bio <- preProcess(training_bio, method = c("center", "scale", "BoxCox"))
training_bio <- predict(preProc_bio, training_bio)

preProc_chem <- preProcess(training_chem, method = c("center", "scale", "BoxCox"))
training_chem <- predict(preProc_chem, training_chem)

rm(corr_bio, corr_chem)  # keep clean workspace

# 5. Prep test data sets (following training alterations)
temp <- colnames(test_bio) %in% colnames(training_bio)
test_bio <- test_bio[ ,temp]
test_bio <- predict(preProc_bio, test_bio)  # same preProc object

temp <- colnames(test_chem) %in% colnames(training_chem)
test_chem <- test_chem[ ,temp]
test_chem <- predict(preProc_chem, test_chem)  # same preProc object

rm(preProc_bio, preProc_chem) 

# Fit linear classification models to optimise AUC ============================

# LDA -------------------------------------------------------------------------
# Briefly: background to linear discriminant analysis (LDA)...
# LDA models the distribution of the predictors X given the response Y (with k
# classes) and uses Bayes theorem to flip these around into estimates of 
# Pr(y = k|X = x). Let p(k) represent the overall (prior) probability that any 
# random observation is from the kth class of Y. Let f(x = X|Y = k) denote the
# density function of X for an observation that comes from the kth class. f(x) will
# be high when a given value of X is likely to be associated with class k of Y.
# Then Bayes theorem states Pr(y = k|X = x) = pk(x) = p(k)*fk(x)/SUM[p(l)*fl(x)]. 
# p(k), the posterior probability, can be estimated using sample fractions of 
# classes, however f(x) is more difficult w/out assuming a probability distribution
# for X.
# By modelling these distributions as Gaussian, and assuming equal variances the
# discriminant function can be derived.

# 1a. fit LDA - biological data
lda_bio <- lda(injury[training_index] ~ ., 
  data = training_bio)

# 1a. fit LDA - chem data
# failed - collinearity exists in prediction matrix
# remove collinearity
remove <- findLinearCombos(training_chem)$remove
training_chem <- training_chem[ ,-remove]
temp <- colnames(test_chem) %in% colnames(training_chem)
test_chem <- test_chem[ ,temp]

lda_chem <- lda(injury[training_index] ~ ., 
  data = training_chem)

# 2a. predict on test data - bio
lda_bio_preds <- predict(lda_bio, test_bio)
confusionMatrix(lda_bio_preds$class, injury[-training_index])

# 2b. predict on test data - chem
lda_chem_preds <- predict(lda_chem, test_chem)
confusionMatrix(lda_chem_preds$class, injury[-training_index])




