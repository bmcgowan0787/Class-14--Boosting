
###########################
## Chapter 8 Exercise 10 ##
###########################

# a) Remove the observations for whom the salary information is unknown, and then log-transform the salaries.

# b) Create a training set consisting of the first 200 observations, and
# a test set consisting of the remaining observations.

# c) Perform boosting on the training set with 1,000 trees for a range
# of values of the shrinkage parameter ??. Produce a plot with
# different shrinkage values on the x-axis and the corresponding
# training set MSE on the y-axis.

# d) Produce a plot with different shrinkage values on the x-axis and
# the corresponding test set MSE on the y-axis.

# e) Compare the test MSE of boosting to the test MSE that results
# from applying two of the regression approaches seen in Chapters 3 and 6.
# Note: We used lasso from Ch. 6

# f) Which variables appear to be the most important predictors in
# the boosted model?

# g) Now apply bagging to the training set. What is the test set MSE
# for this approach?

###########################
### Answers to Ex. 8.10 ###
###########################

# Remove observation where salary is unknown, and log transform salaries
library(ISLR)
sum(is.na(Hitters$Salary))
Hitters = Hitters[-which(is.na(Hitters$Salary)), ] # Removes NAs
sum(is.na(Hitters$Salary))
Hitters$Salary = log(Hitters$Salary) # Log Transform

# Create a training set for the first 200 observations, and a test of the remaining
train = 1:200
Hitters.train = Hitters[train, ]
Hitters.test = Hitters[-train, ]

# Perform boosting on traiing set with 1,000 trees for a range of values of the shrinkage parameter
library(gbm)
set.seed(103)
pows = seq(-10, -0.2, by = 0.1)
lambdas = 10^pows
length.lambdas = length(lambdas)
train.errors = rep(NA, length.lambdas)
test.errors = rep(NA, length.lambdas)
for (i in 1:length.lambdas) {
  boost.hitters = gbm(Salary ~ ., data = Hitters.train, distribution = "gaussian", 
                      n.trees = 1000, shrinkage = lambdas[i])
  train.pred = predict(boost.hitters, Hitters.train, n.trees = 1000)
  test.pred = predict(boost.hitters, Hitters.test, n.trees = 1000)
  train.errors[i] = mean((Hitters.train$Salary - train.pred)^2)
  test.errors[i] = mean((Hitters.test$Salary - test.pred)^2)
}

# Produce a plot with different shrinkage values on the x-axis
# and plot the corresponding training set MSE on the y-axis
plot(lambdas, train.errors, type = "b", xlab = "Shrinkage", ylab = "Train MSE", 
     col = "blue", pch = 20)

# Produce a plot of shrinkage vs. test MSE
plot(lambdas, test.errors, type = "b", xlab = "Shrinkage", ylab = "Test MSE", 
     col = "red", pch = 20)
min(test.errors)
lambdas[which.min(test.errors)]
# Min test error at ??=0.05

# Compare Boosting test MSE to test MSE from regression approaches in Ch. 3 & 6
lm.fit = lm(Salary ~ ., data = Hitters.train)
lm.pred = predict(lm.fit, Hitters.test)
mean((Hitters.test$Salary - lm.pred)^2)

# install.packages("glmnet")
library(glmnet)

set.seed(134)
x = model.matrix(Salary ~ ., data = Hitters.train)
y = Hitters.train$Salary
x.test = model.matrix(Salary ~ ., data = Hitters.test)
lasso.fit = glmnet(x, y, alpha = 1)
lasso.pred = predict(lasso.fit, s = 0.01, newx = x.test)
mean((Hitters.test$Salary - lasso.pred)^2)