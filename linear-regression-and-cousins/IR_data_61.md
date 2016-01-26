# Ex6.1 - Linear regression: IR spectrum of food
Oisin Fitzgerald  
26 January 2016  
### The data:
The data provides an infrared (IR) profile and analytical chemistry determined 
percent content of water, fat, and protein for meat samples. If there can be establish 
a predictive relationship between IR spectrum and fat content, then food scientists 
could predict a sample’s fat content with IR instead of using analytical chemistry

### Outline:
1. What is the relationship between the predictors? Are they highly correlated given 
the same food sample is measured at many IR wavelengths?
2. Create training/test split
3. Fit different models
  + Linear regression
  + Ridge regression, lasso and elastic net
  + PCR and PLS
4. Compare models predictive ability


```r
# load data and packages
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(car)
library(corrplot)
library(elasticnet)
```

```
## Loading required package: lars
```

```
## Loaded lars 1.2
```

```r
library(lars)
library(broom)
library(reshape2)
library(pls)
```

```
## 
## Attaching package: 'pls'
```

```
## The following object is masked from 'package:corrplot':
## 
##     corrplot
```

```
## The following object is masked from 'package:caret':
## 
##     R2
```

```
## The following object is masked from 'package:stats':
## 
##     loadings
```

```r
data(tecator) # from caret
```

### 1. Relationship between predictors and distributions 


```r
# correlation
XX <- cor(absorp)
XX[1:5, 1:5]  # everything is related to everything!
```

```
##           [,1]      [,2]      [,3]      [,4]      [,5]
## [1,] 1.0000000 0.9999908 0.9999649 0.9999243 0.9998715
## [2,] 0.9999908 1.0000000 0.9999916 0.9999678 0.9999309
## [3,] 0.9999649 0.9999916 1.0000000 0.9999923 0.9999707
## [4,] 0.9999243 0.9999678 0.9999923 1.0000000 0.9999930
## [5,] 0.9998715 0.9999309 0.9999707 0.9999930 1.0000000
```

```r
# PCA
pca_object <- prcomp(absorp)
percent_variance <- pca_object$sdev^2/sum(pca_object$sd^2)*100
head(percent_variance)
```

```
## [1] 98.679162750  0.900926147  0.296292185  0.114005307  0.005754017
## [6]  0.002516023
```

```r
# Predictor distributions
ggplot(data = data.frame(absorp)) + 
  geom_histogram(aes(x = X1), bins = 20, col = 1) +
  labs(title = "Histogram of IR wavelength no. 1",
    x = "Wavelength predictor 1")  # positive skew
```

![](IR_data_61_files/figure-html/unnamed-chunk-2-1.png)\

### 2. Create a training/test split

* 75% of the data to the training set
* As the predictors were positively skewed the Yeo-Johnson transformation procedure 
was carried out. The lambda of transformation was approximately -1 for all predictors.


```r
length(endpoints[ ,1])  # how many observations?
```

```
## [1] 215
```

```r
# create partition index
data_split <- createDataPartition(endpoints[ ,1], p = .75)
data_split <- data_split$Resample1

# split data
training <- absorp[data_split, ]
test <- absorp[-data_split, ]
train_resp <- endpoints[data_split, 2]  # column 2 is fat content
test_resp <- endpoints[-data_split, 2]

# de-skew variables
proc_object <- preProcess(training, 
  method = c("center", "scale", "YeoJohnson"))
```

```
## Warning in pre_process_options(method, column_types): The following pre-
## processing methods were eliminated: 'center', 'scale', 'YeoJohnson'
```

```r
training <- predict(proc_object, training)
training <- data.frame(training)
test <- predict(proc_object, test)
test <- data.frame(test)
```

### 3. Model fitting
* Linear regression  
    + Unsurprisingly prior removing of highly correlated predictors resulted in a model
    with only one independent variable. The performance on cross-validation was poor.
* Ridge regression
    + The ridge model quickly highlighted the ability to improve on the linear regression
    model. However, subsequent fitting of a lasso model showed that an ability to drive
    the coefficients to zero was an advantage in the highly correlated predictor environment.
* The lasso and elastic net
    + As noted the lasso model outperformed the ridge model. The optimal solution resulted
    in a large number of the coefficient being shrunk to zero
    + Enet performed similar to the lasso, with the best performing model having a
    low lambda for the ridge function  
* Principal components and partial least squares regression
    + These both performed quite well. The similarity of the PCR model to the PLS
    models is likely related to the variance in the predictors (IR response) very much 
    being a consequence of the variance in the response (food fat content), thus the 
    unsupervised nature of PCA causing little detriment.
    + The number of principal components was tuned rather than using the first two,
    or fist few that explained 90% of variance etc.


```r
ctrl <- trainControl(method = "cv", number = 5, repeats = 5)
# Linear regression
mc <- findCorrelation(training, cutoff = 0.95)
training_linear <- data.frame(training[ ,-mc])
colnames(training_linear) <- "X1"
linear_model <- train(y = train_resp,
  x = training_linear,
  method = "lm",
  trControl = ctrl)
linear_model
```

```
## Linear Regression 
## 
## 163 samples
##   1 predictor
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 130, 131, 131, 131, 129 
## Resampling results
## 
##   RMSE      Rsquared   RMSE SD   Rsquared SD
##   11.81486  0.1850964  1.169411  0.12459    
## 
## 
```

```r
# Ridge Regression - penalise square of coefficient
ridge_model <- train(y = train_resp,
  x = training,
  method = "ridge",
  trControl = ctrl,
  tuneLength = 10)
ridge_model
```

```
## Ridge Regression 
## 
## 163 samples
## 100 predictors
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 131, 131, 129, 131, 130 
## Resampling results across tuning parameters:
## 
##   lambda        RMSE      Rsquared   RMSE SD    Rsquared SD
##   0.0000000000  6.685099  0.7838891  2.2209448  0.10196157 
##   0.0001000000  2.991488  0.9493314  0.4811632  0.02811323 
##   0.0002371374  3.063830  0.9470324  0.4727954  0.02985508 
##   0.0005623413  3.125773  0.9449306  0.4797607  0.03155044 
##   0.0013335214  3.205681  0.9420948  0.5087917  0.03410199 
##   0.0031622777  3.373140  0.9359861  0.5474163  0.03781703 
##   0.0074989421  3.681126  0.9238773  0.6030849  0.04272802 
##   0.0177827941  4.083951  0.9066681  0.6934984  0.04978018 
##   0.0421696503  4.598176  0.8842018  0.7629404  0.06105642 
##   0.1000000000  5.437224  0.8423472  0.7278904  0.07869157 
## 
## RMSE was used to select the optimal model using  the smallest value.
## The final value used for the model was lambda = 1e-04.
```

```r
plot(ridge_model)
```

![](IR_data_61_files/figure-html/unnamed-chunk-4-1.png)\

```r
# Lasso - penalise absolute value of coeffienct
lasso_grid <- expand.grid(.fraction = seq(0.001, 0.1, 0.01))
lasso_model <- train(y = train_resp,
  x = training,
  method = "lasso",
  trControl = ctrl,
  tuneGrid = lasso_grid)
lasso_model
```

```
## The lasso 
## 
## 163 samples
## 100 predictors
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 131, 130, 131, 130, 130 
## Resampling results across tuning parameters:
## 
##   fraction  RMSE      Rsquared   RMSE SD    Rsquared SD
##   0.001     2.869473  0.9539881  0.3557413  0.006615380
##   0.011     2.426025  0.9650132  0.2820628  0.007243307
##   0.021     2.695235  0.9561605  0.4593137  0.014794485
##   0.031     2.911562  0.9492599  0.5398003  0.017747837
##   0.041     3.048880  0.9446928  0.6154857  0.019349598
##   0.051     3.155280  0.9406554  0.7200859  0.022670036
##   0.061     3.205895  0.9381256  0.8888294  0.028597190
##   0.071     3.245370  0.9360626  1.0646941  0.035339960
##   0.081     3.265245  0.9346535  1.2016616  0.041247213
##   0.091     3.295686  0.9324666  1.3522449  0.048174173
## 
## RMSE was used to select the optimal model using  the smallest value.
## The final value used for the model was fraction = 0.011.
```

```r
plot(lasso_model)
```

![](IR_data_61_files/figure-html/unnamed-chunk-4-2.png)\

```r
# Elastic Net - combination of ridge and lasso
enet_grid <- expand.grid(.fraction = seq(0.001, 0.1, 0.01), .lambda = c(0, 0.0001, 0.001, 0.01))
enet_model <- train(y = train_resp,
  x = training,
  method = "enet",
  trControl = ctrl,
  tuneGrid = enet_grid)
enet_model
```

```
## Elasticnet 
## 
## 163 samples
## 100 predictors
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 129, 131, 131, 131, 130 
## Resampling results across tuning parameters:
## 
##   lambda  fraction  RMSE       Rsquared   RMSE SD    Rsquared SD
##   0e+00   0.001      3.031717  0.9487117  0.8187863  0.02591673 
##   0e+00   0.011      2.228923  0.9701910  0.3515265  0.01218490 
##   0e+00   0.021      2.333464  0.9659619  0.4698750  0.01785094 
##   0e+00   0.031      2.551397  0.9589941  0.6694096  0.02193653 
##   0e+00   0.041      2.782969  0.9508523  0.9363681  0.03211693 
##   0e+00   0.051      2.928699  0.9453797  1.1289957  0.04097525 
##   0e+00   0.061      2.944859  0.9455735  1.1169394  0.04029928 
##   0e+00   0.071      3.002993  0.9433680  1.2074149  0.04416056 
##   0e+00   0.081      2.989357  0.9447093  1.1429715  0.04056460 
##   0e+00   0.091      2.973706  0.9466711  1.0303305  0.03464044 
##   1e-04   0.001     12.475360  0.3039583  0.8718890  0.18268391 
##   1e-04   0.011     10.734017  0.3152900  1.5227099  0.19085746 
##   1e-04   0.021     10.200875  0.3897025  1.4592485  0.18485671 
##   1e-04   0.031      9.677944  0.4640394  1.3948727  0.17095603 
##   1e-04   0.041      9.166785  0.5345581  1.3290572  0.15183186 
##   1e-04   0.051      8.671201  0.5982012  1.2591961  0.13013645 
##   1e-04   0.061      8.194427  0.6531280  1.1865073  0.10862984 
##   1e-04   0.071      7.736124  0.6992557  1.1132828  0.08932306 
##   1e-04   0.081      7.297789  0.7373912  1.0380371  0.07288301 
##   1e-04   0.091      6.898631  0.7676460  0.9363783  0.05882363 
##   1e-03   0.001     12.636241  0.3039583  0.8870148  0.18268391 
##   1e-03   0.011     11.178478  0.3039583  1.1738458  0.18268391 
##   1e-03   0.021     10.718328  0.3173703  1.5052553  0.18848532 
##   1e-03   0.031     10.431307  0.3572448  1.4630710  0.18512760 
##   1e-03   0.041     10.147270  0.3976560  1.4206121  0.17928836 
##   1e-03   0.051      9.866361  0.4379708  1.3783684  0.17135944 
##   1e-03   0.061      9.588549  0.4776096  1.3360788  0.16176252 
##   1e-03   0.071      9.313993  0.5160096  1.2935612  0.15093023 
##   1e-03   0.081      9.042944  0.5526916  1.2509801  0.13935801 
##   1e-03   0.091      8.775811  0.5872593  1.2085864  0.12750272 
##   1e-02   0.001     12.719429  0.3039583  0.8918002  0.18268391 
##   1e-02   0.011     11.803486  0.3036823  0.9267667  0.18258930 
##   1e-02   0.021     11.126023  0.3035693  1.2041062  0.18256740 
##   1e-02   0.031     10.792840  0.3049571  1.4600966  0.18531782 
##   1e-02   0.041     10.683744  0.3211450  1.5018459  0.18893556 
##   1e-02   0.051     10.528113  0.3424784  1.4780302  0.18747353 
##   1e-02   0.061     10.373441  0.3640162  1.4540161  0.18518625 
##   1e-02   0.071     10.219559  0.3856985  1.4301241  0.18218421 
##   1e-02   0.081     10.067041  0.4073343  1.4057774  0.17846760 
##   1e-02   0.091      9.915991  0.4288132  1.3816928  0.17421012 
## 
## RMSE was used to select the optimal model using  the smallest value.
## The final values used for the model were fraction = 0.011 and lambda = 0.
```

```r
plot(enet_model)
```

![](IR_data_61_files/figure-html/unnamed-chunk-4-3.png)\

```r
# PCR - 
pcr_results <- list(results = data.frame(RMSE = NA, RMSE_sd = NA), final = NA)
for (i in 1:20) {
  # fit model
  train_data <- princomp(training)$scores[ ,1:i]
  train_data <- data.frame(train_data)
  pcr_model <- train(y = train_resp,
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
```

```
## $results
##         RMSE   RMSE_sd
## 1  11.224138 1.7576017
## 2  11.236943 1.1890817
## 3   8.167315 1.1748668
## 4   4.386462 0.5776326
## 5   3.459017 0.4443272
## 6   3.226676 0.2006224
## 7   3.223908 0.4853659
## 8   3.134869 0.3667502
## 9   3.140171 0.7163722
## 10  3.094634 0.4911017
## 11  3.095387 0.5346804
## 12  2.860944 0.4821079
## 13  2.913053 0.4374026
## 14  3.245235 0.9542605
## 15  3.205204 0.8763168
## 16  2.972045 0.7732798
## 17  3.096274 1.3275177
## 18  3.124324 0.8857930
## 19  3.093565 0.8149503
## 20  3.056621 0.4924618
## 
## $final
## Linear Regression 
## 
## 163 samples
##  12 predictor
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 130, 129, 131, 131, 131 
## Resampling results
## 
##   RMSE      Rsquared   RMSE SD    Rsquared SD
##   2.860944  0.9533487  0.4821079  0.01421811 
## 
## 
```

```r
# PLS
pls_grid <- expand.grid(.ncomp = seq(10, 20, 1))
pls_model <- train(y = train_resp,
  x = training,
  method = "pls",
  trControl = ctrl,
  preProcess = c("center", "scale"),
  tuneGrid = pls_grid)
pls_model
```

```
## Partial Least Squares 
## 
## 163 samples
## 100 predictors
## 
## Pre-processing: centered (100), scaled (100) 
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 130, 131, 131, 130, 130 
## Resampling results across tuning parameters:
## 
##   ncomp  RMSE      Rsquared   RMSE SD    Rsquared SD
##   10     2.813184  0.9554810  0.4550704  0.01689361 
##   11     2.737955  0.9561945  0.5293079  0.01889828 
##   12     2.630719  0.9588998  0.5571360  0.01612485 
##   13     2.499763  0.9638177  0.4054965  0.01162614 
##   14     2.509081  0.9638480  0.4850952  0.01039794 
##   15     2.638470  0.9610913  0.5039031  0.01150921 
##   16     2.825153  0.9571256  0.5796790  0.01380110 
##   17     2.877523  0.9562638  0.6640526  0.01825556 
##   18     2.800149  0.9573154  0.6760719  0.01987246 
##   19     2.852297  0.9557518  0.7011687  0.02034438 
##   20     2.926403  0.9521821  0.8639022  0.02395220 
## 
## RMSE was used to select the optimal model using  the smallest value.
## The final value used for the model was ncomp = 13.
```

### 4. Compare performance on test set
* The results from fitting on the test set followed from the cross-validation 
included in model fitting. Linear regression did very poorly with ridge regression 
slightly worse off than the group of lasso, elastic net, PCR and PLS. 
* In context the RMSE and correlation between predicted and observed results are 
superb, and surely suggest that any of these models could be used in measuring the
fat content of food using infrared.
* Given the similarities in the model performances I was interested in constructing
confidence intervals around the RMSE. A function to calculate bootstrap 
estimation and its results are shown below.


```r
# Linear regression
test_linear <- data.frame(test[ ,-mc])
colnames(test_linear) <- colnames(training_linear)
linear_pred <- predict(linear_model, test_linear)
ggplot() + geom_point(aes(x = linear_pred, y = test_resp))
```

![](IR_data_61_files/figure-html/unnamed-chunk-5-1.png)\

```r
n <- length(test_resp)
RMSE_lm <- sqrt(sum((test_resp - linear_pred)^2)/n); RMSE_lm
```

```
## [1] 12.0309
```

```r
# Ridge regression
ridge_preds <- predict(ridge_model, test)
ggplot() + geom_point(aes(x = ridge_preds, y = test_resp))
```

![](IR_data_61_files/figure-html/unnamed-chunk-5-2.png)\

```r
RMSE_ridge <- sqrt(sum((test_resp - ridge_preds)^2)/n); RMSE_ridge
```

```
## [1] 2.562405
```

```r
# Lasso
lasso_preds <- predict(lasso_model, test)
ggplot() + geom_point(aes(x = lasso_preds, y = test_resp))
```

![](IR_data_61_files/figure-html/unnamed-chunk-5-3.png)\

```r
RMSE_lasso <- sqrt(sum((test_resp - lasso_preds)^2)/n); RMSE_lasso
```

```
## [1] 1.906485
```

```r
# Elastic net
enet_preds <- predict(enet_model, test)
ggplot() + geom_point(aes(x = enet_preds, y = test_resp))
```

![](IR_data_61_files/figure-html/unnamed-chunk-5-4.png)\

```r
RMSE_enet <- sqrt(sum((test_resp - enet_preds)^2)/n); RMSE_enet
```

```
## [1] 1.906485
```

```r
# PCR
pca_train <- princomp(training)
test_pcs <- predict(pca_train, test)
pcr_preds <- predict(pcr_results$final, test_pcs)
ggplot() + geom_point(aes(x = pcr_preds, y = test_resp))
```

![](IR_data_61_files/figure-html/unnamed-chunk-5-5.png)\

```r
RMSE_pcr <- sqrt(sum((test_resp - pcr_preds)^2)/n); RMSE_pcr
```

```
## [1] 2.441669
```

```r
# PLS
pls_preds <- predict(pls_model, test)
ggplot() + geom_point(aes(x = pls_preds, y = test_resp))
```

![](IR_data_61_files/figure-html/unnamed-chunk-5-6.png)\

```r
RMSE_pls <- sqrt(sum((test_resp - pls_preds)^2)/n); RMSE_pls
```

```
## [1] 2.166212
```

```r
cor(pls_preds, test_resp)
```

```
## [1] 0.9850384
```

### 4. Compare performance on test set contd. 
* Bootstrap estimate of RMSE confidence interval
    + PLS appears to be the prefered model, it shows the least variation in its RMSE scores
    across the bootstrap samples. PCR is likely similar, an issue with variable naming meant
    I excluded it.


```r
boostrap_RMSE <- function(model, data, obs, trials = 1000, CI = 0.95) {
  
  n <- nrow(data)
  out <- list(results = data.frame(RMSE = NA), lower = NA, upper = NA)
  
  for (i in 1:trials) {
    # create bootstrap sample
    samp <- sample(n, size = n, replace = TRUE)
    boot_obs <- obs[samp]
    boot_data <- data.frame(data[samp, ])
    colnames(boot_data) <- colnames(data)
    # predict
    preds <- predict(model, newdata = boot_data)
    RMSE <- sqrt(sum((boot_obs - preds)^2)/n)
    
    out$results[i ,1] <- RMSE
  }
  
  temp <- out$results$RMSE
  temp <- quantile(temp, probs = c(0.025, 0.975), na.rm = TRUE)
  
  out$lower <- temp[1]
  out$upper <- temp[2]
  
  out
}
```



```r
# The bootstrap results
bRMSE_lm <- boostrap_RMSE(linear_model, test_linear, test_resp)
bRMSE_ridge <- boostrap_RMSE(ridge_model, test, test_resp)
bRMSE_lasso <- boostrap_RMSE(lasso_model, test, test_resp)
bRMSE_enet <- boostrap_RMSE(enet_model, test, test_resp)
# bRMSE_pcr <- boostrap_RMSE(pcr_model, test, test_resp)
bRMSE_pls <- boostrap_RMSE(pls_model, test, test_resp)

model_results <- data.frame(bRMSE_lm$results, bRMSE_ridge$results,
  bRMSE_lasso$results, bRMSE_enet$results, bRMSE_pls$results)
colnames(model_results) <- c('lm', 'ridge', 'lasso', 'enet', 'pls')

temp <- melt(model_results)
```

```
## No id variables; using all as measure variables
```

```r
ggplot(data = temp, aes(x = variable, y = value)) + 
  geom_boxplot(width = 0.5) + 
  theme_bw() + 
  labs(title = 'Bootstrap Estimates of Model Performance',
    x = 'Model',
    y = 'RMSE')
```

![](IR_data_61_files/figure-html/unnamed-chunk-7-1.png)\

### Conclusion

* The clear signal in the data meant that despite multicollinarity issues several 
linear model fitting methods had no problem producing extremely predictive models. 
* The predictors were highly correlated and likely possessed variations on same 
information. Therefore possibly as a result of their ability to extract the minimal 
dimension signal from several correlated variables PCR and PLS appear to have a slight 
performance advantage over other models.
