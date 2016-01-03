# Classification Trees and Rule Based Models
#
# Script based on Ex 14.2 of Applied Predictive Modeling (Kuhn & Johnson, 2013)
#
# AIM: predict whether a customer is likely to leave a telecomm company based 
# on 19 variables
#
# In particular answering:
# a) Impact of grouped vs. binary dummy encoding of the state categorical 
# variable
# b) Which classification technique works best of (gini fit) classificaiton 
# trees, PART and C50 rule based model and bagged trees
#
# Load data and packages ======================================================
library(C50)
library(caret)
library(rpart)
library(partykit)
library(tree)
library(pROC)
library(ROCR)
library(randomForest)
library(RWeka)

data(churn)  # load data

str(churnTrain)
head(churnTrain)
table(churnTrain$churn)  # outcome variable category frequencies
prop.table(table(churnTrain$churn)) # outcome variable category probs

# Basic tree models ===========================================================
# Classification tree model
tree_model <- rpart(churn ~ ., data = churnTrain)  # classification tree
print(tree_model)  # decision nodes/splits

# Visualise unpruned tree (default method)
plot(tree_model, branch = 0.3, compress = TRUE, xpd = NA)  # plot the decision tree
text(tree_model, use.n = TRUE, cex = 0.7)  # nodes info

# Alternative visualisation no. 1
party_tree <- as.party(tree_model)
plot(party_tree, type = "simple", gp = gpar(cex = 0.5))

# Alternative visualisation no. 2 (not working - too many states?)
# draw.tree(party_tree, cex = 0.5, nodeinfo = TRUE, col = (0:8 / 8))


# Prune tree model ------------------------------------------------------------
plotcp(tree_model)  # plot of complexity parameter info
printcp(tree_model)  # variables used and complexity parameter info
# from printcp the optimal tree size is 8 leaves (cp = 0.035)
tree_model_final <- prune(tree_model, cp = 0.035)

# Visualise final (pruned) model
party_tree_final <- as.party(tree_model_final)
plot(party_tree_final, type = "simple", gp = gpar(cex = 0.6))
plot(party_tree_final, gp = gpar(cex = 0.6))  # nice plot of the tree


# Evaluate on the test data ---------------------------------------------------
tree_model_pred <- predict(tree_model_final, churnTest)  # make predictions
temp <- ifelse(tree_model_pred[ ,1] < .5, "no", "yes")
tree_model_pred <- data.frame(tree_model_pred, prediction = as.factor(temp))
tree_confusion <- table(tree_model_pred$prediction, churnTest$churn)
tree_confusion  # print confusion matrix
prop.table(tree_confusion)
(141+1415)/1667  # 93.34% of observations are correctly classified

# ROC curve and AUC with pROC package
roc_tree <- roc(response = churnTest$churn, 
  predictor = tree_model_pred$yes, plot = TRUE)  # ROC curve (sens/spec)
plot(smooth(roc_tree), identity = FALSE)
abline(a = c(1, 1), b = -1, col = 'red', lty = 2)  # add 45' line (confusing param)
ci.sp(roc_tree)  # CI on specificity
ci.se(roc_tree)  # CI on sensitivity

# ROC curve with the ROCR package
rocplot <- function(pred, truth, ...) { # function to create ROC curve
  predob <- prediction(pred, truth )
  perf <- performance(predob, "tpr", "fpr")
  plot(perf, ...)
}
par(pty="s")  # ensure square graph
rocplot(tree_model_pred$yes, churnTest$churn)
abline(a = c(0,0), b = 1, col = 'red', lty = 2)

# AUC
auc(churnTest$churn, tree_model_pred$yes)


# Classification tree using caret:::train --------------------------------------
ctrl <- trainControl(method = "cv", number = 10, classProbs = TRUE)  # 10 fold cv

caret_tree <- train(x = churnTrain[ ,1:19], y = churnTrain$churn,  # note x, y entry
  method = 'rpart', 
  tuneLength = 30, 
  trControl = ctrl)
caret_tree

# Visualise pruned caret tree
caret_tree_party <- as.party(caret_tree$finalModel)
plot(caret_tree_party, gp = gpar(cex = 0.5))  # nice plot of the tree


# Evaluate caret tree on test set ---------------------------------------------
caret_tree_preds <- predict(caret_tree, churnTest, type = "prob")
temp <- ifelse(caret_tree_preds[ ,1] < .5, "no", "yes")
caret_tree_preds <- data.frame(caret_tree_preds, prediction = as.factor(temp))
caret_tree_confusion <- table(caret_tree_preds$prediction, churnTest$churn)
caret_tree_confusion
prop.table(caret_tree_confusion)
(138+1426)/1667  # 93.38% of observations are correctly classified

# ROC curve for caret and tree predictions
rocplot(tree_model_pred$yes, churnTest$churn)
rocplot(caret_tree_preds$yes, churnTest$churn, add = TRUE, col = "blue")
abline(a = c(0,0), b = 1, col = 'red', lty = 2)
legend("bottomright", c("caret model", "rpart model"), 
  lty = 1,
  col = c("blue", "black"), 
  cex = 0.75)

# Slight difference in predictive accuracy of the caret and rpart models.
# The caret model was pruned using accuracy, so this may be the reason.

# Category encoding ===========================================================
# A look at independent vs. grouped categories for the state variable
# i.e. create (levels - 1) dummy variables for state and refit the decision
# trees

# Create dummy variables and add to the (new) dataframe(s)
dmy <- dummyVars( ~ state , data = churnTrain, fullRank = TRUE)
churnTrain_dmy <- data.frame(churnTrain[ ,-1], (predict(dmy, churnTrain)))
churnTest_dmy <- data.frame(churnTest[ ,-1], (predict(dmy, churnTest)))

# Fit and prune a decision tree using caret
str(churnTrain_dmy)
caret_dmytree <- train(x = churnTrain_dmy[ ,-19], y = churnTrain_dmy$churn, 
  method = 'rpart', 
  tuneLength = 30, 
  trControl = ctrl, 
  metric = "Accuracy")
caret_dmytree

# Visualise pruned dummy variable model
caret_dmytree_party <- as.party(caret_dmytree$finalModel)
plot(caret_dmytree_party, gp = gpar(cex = 0.5))  # nice plot of the tree


# Evaluate dummy encoded caret tree on test set --------------------------------
caret_dmytree_preds <- predict(caret_dmytree, churnTest_dmy, type = "prob")
temp <- ifelse(caret_dmytree_preds[ ,1] < .5, "no", "yes")
caret_dmytree_preds <- data.frame(caret_dmytree_preds, prediction = as.factor(temp))
caret_dmytree_confusion <- table(caret_dmytree_preds$prediction, churnTest$churn)
caret_dmytree_confusion
prop.table(caret_tree_confusion)
(142+1431)/1667  # 94.36% of observations are correctly classified

# ROC curve for caret, rpart, dmy var predictions
# Predict probs for the dummy variable model
rocplot(tree_model_pred$yes, churnTest$churn)  # Plot ROC curves
rocplot(caret_tree_preds$yes, churnTest$churn, add = TRUE, col = "blue")
rocplot(caret_dmytree_preds$yes, churnTest$churn, add = TRUE, col = "orange")
abline(a = c(0,0), b = 1, col = 'red', lty = 2)
legend("bottomright", c("caret model", "rpart model", "dummy var model"), lty = 1, 
  col = c("blue", "black", "orange"), cex = 0.65)

# AUC
auc(churnTest$churn, caret_dmytree_preds$yes)


# Bagged tree model ============================================================
# Fit bagged decision tree
# Random forest with n = p is equivalent to bagging
bagged_tree <- randomForest(x = churnTrain[ ,1:19], y = churnTrain$churn,
  importance = TRUE)
bagged_tree


# Using caret ------------------------------------------------------------------
ctrl2 <- trainControl("cv", number = 10)
bagged_tree2 <- train(x = churnTrain[ ,1:19], y = churnTrain$churn, method = "rf",
  importance = TRUE, tuneGrid = expand.grid(mtry = 19), do.trace = TRUE)
bagged_tree2


# Variable importance ----------------------------------------------------------
# MeanDecreaseAccuracy: mean decrease of accuracy in predictions on the out 
# of bag samples when a given variable is excluded from the model.
# MeanDecreaseGini: measure of the total decrease in node impurity that results 
# from splits over that variable, averaged over all trees
varImpPlot(bagged_tree)


# Evaluate bagged decision tree on test data -----------------------------------
bagged_tree2_preds <- predict(bagged_tree2, churnTest, type = "prob")
temp <- ifelse(bagged_tree2_preds[ ,1] < .5, "no", "yes")
bagged_tree2_preds <- data.frame(bagged_tree2_preds, prediction = as.factor(temp))
bagged_tree2_confusion <- table(bagged_tree2_preds$prediction, churnTest$churn)
bagged_tree2_confusion
prop.table(bagged_tree2_confusion)
(188+1320)/1667  # 90.46% accuracy

# ROC curve
par(pty="s")  # ensure square graph
rocplot(tree_model_pred$yes, churnTest$churn)  # Plot ROC curves
rocplot(caret_tree_preds$yes, churnTest$churn, add = TRUE, col = "blue")
rocplot(caret_dmytree_preds$yes, churnTest$churn, add = TRUE, col = "orange")
rocplot(bagged_tree2_preds$yes, churnTest$churn, add = TRUE, col = "green")
abline(a = c(0,0), b = 1, col = 'red', lty = 2)
legend("bottomright", c("caret model", "rpart model", "dummy var model", 
  "bagged tree model"), 
  lty = 1, 
  col = c("blue", "black", "orange", "green"), 
  cex = 0.65)

# AUC
auc(churnTest$churn, bagged_tree2_preds$yes)


# Why is the bagged model less accurate but has a higher AUC ?? ----------------
# See the histograms...
histogram(tree_model_pred$yes)
histogram(caret_dmytree_preds$yes)
histogram(caret_tree_preds$yes)
histogram(bagged_tree2_preds$yes)


# Rule based models ============================================================
# Fit PART model
part_model <- PART(churn ~ ., data = churnTrain)
part_model


# caret method -----------------------------------------------------------------
part_model.c <- train(x = churnTrain[ ,-20], y = churnTrain$churn, 
  method = 'PART', 
  tuneLength = 30, 
  trControl = ctrl, 
  metric = "Accuracy")
part_model.c


# Evaluate bagged classification tree on test data ----------------------------
part_model_preds <- predict(part_model.c, churnTest, type = "prob")
temp <- ifelse(part_model_preds[ ,1] < .5, "no", "yes")
part_model_preds <- data.frame(part_model_preds, prediction = as.factor(temp))
part_model_confusion <- table(part_model_preds$prediction, churnTest$churn)
part_model_confusion
round(prop.table(part_model_confusion), 3)
(130+1376)/1667  # 90.34% accuracy

# ROC curve
par(pty="s")  # ensure square graph
rocplot(tree_model_pred$yes, churnTest$churn)  # Plot ROC curves
rocplot(caret_dmytree_preds$yes, churnTest$churn, add = TRUE, col = "orange")
rocplot(bagged_tree2_preds$yes, churnTest$churn, add = TRUE, col = "green")
rocplot(part_model_preds$yes, churnTest$churn, add = TRUE, col = "blue")
abline(a = c(0,0), b = 1, col = 'red', lty = 2)
legend("bottomright", c("PART rule model", "rpart model", "dummy var model", 
  "bagged tree model"), 
  lty = 1, 
  col = c("blue", "black", "orange", "green"), 
  cex = 0.65)

# AUC
auc(churnTest$churn, part_model_preds$yes)


# Fit C5.0 rule model ---------------------------------------------------------
# information theory based approach, rule voting method
C5.0_model <- C5.0(x = churnTrain[ ,-20], y = churnTrain$churn, rules = TRUE)
C5.0_model
summary(C5.0_model)

# caret version
C5.0_model.c <- train(x = churnTrain[ ,-20], y = churnTrain$churn, 
  method = "C5.0Rules", 
  trControl = ctrl)
C5.0_model.c


# Evaulate C5.0 model on test data --------------------------------------------
C5.0_model_preds <- predict(C5.0_model.c, churnTest, type = "prob")
temp <- ifelse(C5.0_model_preds[ ,1] < .5, "no", "yes")
C5.0_model_preds <- data.frame(C5.0_model_preds, prediction = as.factor(temp))
C5.0_model_confusion <- table(C5.0_model_preds$prediction, churnTest$churn)
C5.0_model_confusion
round(prop.table(C5.0_model_confusion), 3)
(149+1428)/1667  # 94.60% accuracy

# ROC curve
par(pty="s")  # ensure square graph
rocplot(tree_model_pred$yes, churnTest$churn)  # Plot ROC curves
rocplot(caret_dmytree_preds$yes, churnTest$churn, add = TRUE, col = "orange")
rocplot(bagged_tree2_preds$yes, churnTest$churn, add = TRUE, col = "green")
rocplot(part_model_preds$yes, churnTest$churn, add = TRUE, col = "blue")
rocplot(C5.0_model_preds$yes, churnTest$churn, add = TRUE, col = "purple")
abline(a = c(0,0), b = 1, col = 'red', lty = 2)
legend("bottomright", c("PART rule model", "rpart model", "dummy var model", 
  "bagged tree model", "C5.0 rule model"), 
  lty = 1, 
  col = c("blue", "black", "orange", "green", "purple"), 
  cex = 0.65)

# AUC
auc(churnTest$churn, C5.0_model_preds$yes)


# Lift chart ==================================================================
prediction_models <- data.frame(rpart_tree = tree_model_pred$yes, 
  dummy_tree = caret_dmytree_preds$yes,
  bagged_tree = bagged_tree2_preds$yes,
  part_rules = part_model_preds$yes,
  C5.0_rules = C5.0_model_preds$yes)
labs <- c(rpart_tree = "Grouped Categories", dummy_tree = "Binary Categories",
  bagged_tree = "Bagged Tree", 
  part_rules = "PART rules",
  C5.0_rules = "C5.0 rules")
liftCurve <- lift(churnTest$churn ~ rpart_tree + dummy_tree + bagged_tree + 
  part_rules + C5.0_rules, 
  data = prediction_models,
  labels = labs)
xyplot(liftCurve, auto.key = list(columns = 2, lines = TRUE, points = FALSE))


# NB
# rpart, C5.0, and J48 use the formula method differently than
# most other functions by respecting the categorical nature of the data 
# and treating these predictors as grouped sets of categories
# caret:::train function follows the more common convention in R,
# which is to create dummy variables prior to modeling




