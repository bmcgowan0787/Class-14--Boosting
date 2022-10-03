Boosting for Decision Trees
================
Team 5

General Idea
------------

In summary, the idea behind boosting is:

1.  Locate the "weak learners"
2.  Weight them and add them up
3.  Get a stronger predictor

Boosting Methods
----------------

There are various methods of Boosting available. R provides several packages you can use. However, we will only discuss GBM (Gradient Boosting) and ADABoost (Adaptive Boosting) as both are the most popular packages for boosting regression and classification trees, respectively:

1.  **GBM** - known as **gradient boosting.** produces a prediction model in the form of an ensemble of weak prediction models, where the **prediction models are determined iteratively (sequentially)**. Can be used on regression and classification trees. We use it here for regression.

2.  **ADABoost** - an "adaptive learning" model. AdaBoost is adaptive in the sense that subsequent weak learners are tweaked in favor of those instances misclassified by previous classifiers. The output of the other learning algorithms ('weak learners') is combined into a **weighted sum** that represents the final output of the boosted classifier.

Lab 8.3.4 ISLR - Applying Gradient Boosting (GBM)
-------------------------------------------------

### 1.0 - Data Processing

To get started, you will need the "Boston" data set and the "gbm" package for gradient boosting.

``` r
# install.packages("MASS")
# install.packages("gbm")

library(MASS) # Boston data set
library(gbm)
```

Next, We'll develop a Boston training and test set.

-   Set seed to 1 for reproducibility.
-   Use 50% random sample of the Boston data set as a **training set**
-   Remaining variables will be the **test set**

``` r
set.seed(1)
train = sample(1:nrow(Boston), nrow(Boston)/2)
boston.test = Boston[-train,"medv"]
```

### 1.1 Apply Boosting Function

The most common package in R for boosting decision trees is the **"gbm"** package. The functions we will be using from the "gbm" package are simply gbm().

Lets apply gbm to the Boston data set:

``` r
boost.boston = gbm(medv ~.,                   #formula
                   data = Boston[train,],     #training dataset
                   distribution = 'gaussian', #'gaussian' for regression models, 'bernoulli' for classification
                   n.trees = 5000,            #number of trees
                   interaction.depth = 4,     #depth of each tree or number of leaves
                   shrinkage = 0.001          #0.001 default value
                   )
# Recall shrinkage can be 0.1 to 0.001.  The smaller the shrinkage parameter (lambda) the greater the number of trees necessary
```

Let's analyze the boosting effects by using summary().

``` r
summary(boost.boston)
```



    ##             var     rel.inf
    ## lstat     lstat 45.96792013
    ## rm           rm 31.22018272
    ## dis         dis  6.80567724
    ## crim       crim  4.07534048
    ## nox         nox  2.56586166
    ## ptratio ptratio  2.26983216
    ## black     black  1.78740116
    ## age         age  1.64495723
    ## tax         tax  1.36917603
    ## indus     indus  1.27052715
    ## chas       chas  0.80066528
    ## rad         rad  0.20727091
    ## zn           zn  0.01518785

The summary of the boosted data set shows us a **relative influence** plot of the variables and a table with each exact statistics.

### 1.2 Relative Influence

-   **What does relative influence mean?**

What relative influence attempts to convey is the decrease in MSE for each unit change in the predictor. For example, in the graph above, we see that the **lstat** variable has the largest impact on MSE. Stating that for each unit change (decrease in this example) in **lstat**, the MSE decreases by 45.96 for predicting the medv (medivan value) of a home.

<img src="Figures/Relative Influence-1.png"/>

### 1.3 Partial Dependence Plot

Take note that **lstat** and **rm** are the most influential variables in the data set. We can create a **partial dependence plot** for both variables to analyze their effects individually.

These plots illustrate the **marginal effect** (the change in predicted probability for a unit change in the predictor) of the selected variables on the response after integrating out the other variables. In this case, as we might expect, median house prices are increasing with rm and decreasing with lstat.

``` r
par(mfrow=c(1,2))
plot(boost.boston,i="rm", col= 'red', lwd = 2, main = "Average Number of Rooms") 

# "i"" is the x variable used from the boosting model

plot(boost.boston,i="lstat", col = 'blue', lwd = 2, main = "% Lower Status of Population")
```

<img src="Figures/Partial Dependence Plot of Boston-1.png"/>

### 1.4 Predict using the Boosted Model

We can now use the boosted model to predict the median value (**medv**) on the test set.

``` r
yhat.boost <- predict(boost.boston, newdata = Boston[-train,], n.trees = 5000) # Use 5000 trees again for test set
gbm.test.mse <- mean((yhat.boost - boston.test)^2)                
print(gbm.test.mse)
```

    ## [1] 11.84694

-   **Take note of the 11.85 test MSE for later.**

### 1.5 Predicted with larger shrinkage parameter

Can we improve the accuracy of the boosted model and lower the test MSE?

Let's adjust the shrinkage paramater:

``` r
set.seed(1)
boost.boston2 = gbm(medv ~.,                   #formula
                   data = Boston[train,],     #training dataset
                   distribution = 'gaussian', #'gaussian' for regression models, 'bernouli' for classification
                   n.trees = 5000,            #number of trees
                   interaction.depth = 4,     #depth of each tree or number of leaves
                   shrinkage = 0.2,           #Increased param to 0.2
                   verbose = F                #Don't print performance indicators
                   )

yhat.boost2 = predict(boost.boston2 ,newdata = Boston[-train,], n.trees = 5000)
gbm.mse.02 <- mean((yhat.boost2 -boston.test)^2)
```

### 1.6 Compare Test MSEs

Which of the two boosted models has the lowest MSE?

``` r
print(paste("MSE of Boosted Default: ", gbm.test.mse))
```

    ## [1] "MSE of Boosted Default:  11.8469398169183"

``` r
print(paste("MSE of Boosted with .2 Shrinkage: ", gbm.mse.02))
```

    ## [1] "MSE of Boosted with .2 Shrinkage:  12.1405273629917"

Boosting with ADA (Adapative Learning)
--------------------------------------

### 2.0 Data Preprocessing

ADABoost is an **adaptive learning function** and a form of additive logistic regression. However, ada cannot be applied to regression models. It is only applicable to **two-class models** or classification models.

ADABoost is the most desired boosting method for classification. To demonstrate ADA, we will use a classification data set called "churn". The following R packages are necessary for performing ADABoost:

-   **"C50"** - contains the "churn" dataset
-   **"ada"** - for adaptive learning boosting. Will apply the **weighted averages** of each model.
-   **"ggplot2"** - for plotting and comparing results

``` r
# install.packages("C50")
# install.packages("ggplot2")
# install.packages("ada")

library(C50)
library(ggplot2)
library(ada)

data(churn)         #Pulls churnTrain and churnTest data from C50
na.omit(churnTrain) #MUST ensure that all NA values are omitted from the data
na.omit(churnTest)  #Or function will fail
```

### 2.1 Applying ADABoost

Now, let's apply ADABoost to the churn training set:

``` r
set.seed(1)

adafit <- ada(churn ~.,             #formula
              data = churnTrain,    #training data set
              iter = 50,            #number of tree iterations
              bag.frac = 0.5,       #Randomly samples the churnTrain set. Value of 1 equivalent to bagging
              rpart.control(maxdepth=30,minsplit=20,cp=0.01,xval=10)
              )
#maxdepth controls depth of trees (leaves),  minsplit is the minimum number of observations in a node before attempting split (20) and that split must decrease the overall error by 0.01 (cp controls complexity)

print(adafit)
```

    ## Call:
    ## ada(churn ~ ., data = churnTrain, iter = 50, bag.frac = 0.5, 
    ##     rpart.control(maxdepth = 30, minsplit = 20, cp = 0.01, xval = 10))
    ## 
    ## Loss: exponential Method: discrete   Iteration: 50 
    ## 
    ## Final Confusion Matrix for Data:
    ##           Final Prediction
    ## True value   no  yes
    ##        no  2850    0
    ##        yes   84  399
    ## 
    ## Train Error: 0.025 
    ## 
    ## Out-Of-Bag Error:  0.034  iteration= 50 
    ## 
    ## Additional Estimates of number of iterations:
    ## 
    ## train.err1 train.kap1 
    ##         45         45

### 2.2 Plotting the data

Use standard R plot to observe the decreasing training error rate. A series of 5 1's identifies the curve.

``` r
plot(adafit) 
```

<img src="Figures/ADA curve-1.png"/>

Determine the variables of importance.

``` r
varplot(adafit)
```

<img src="Figures/ADA varplot-1.png"/>

### 2.3 Predict

``` r
prtrain <- predict(adafit, newdata=churnTrain)
#table(churnTrain[,"churn"], prtrain , dnn=c("Actual", "Predicted"))
round(100* table(churnTrain$churn, prtrain,dnn=c("% Actual", "% Predicted"))/length(prtrain),1)
```

    ##         % Predicted
    ## % Actual   no  yes
    ##      yes  2.5 12.0
    ##      no  85.5  0.0

``` r
prtrain <- predict(adafit, newdata=churnTest)
#table(churnTrain[,"churn"], prtrain , dnn=c("Actual", "Predicted"))
round(100* table(churnTest$churn, prtrain,dnn=c("% Actual", "% Predicted"))/length(prtrain),1)
```

    ##         % Predicted
    ## % Actual   no  yes
    ##      yes  5.3  8.1
    ##      no  85.5  1.0
