ource("C:/Users/Aashish/techfield/algorithms/lin_reg_GD_regularization/R/LR_L1Regularization.R")

OLS <- function(y, y_hat) 1/(2*length(y))*sum((y-y_hat)^2)

N <- 50


X <- seq(0,20, length.out = N)

y <- 10 + 3*X + rnorm(N, sd = 1.7)

#y[(N-10):(N-7)] <- y[(N-10):(N-7)] + 30

plot(X,y)

X <- matrix(X)

lin_reg <- LinearRegression_GD_L1regularization$new()

lin_reg$fit(X,y,lambda1 = 200)

y_hat <- lin_reg$predict(X)

plot(X,y)
lines(X,y_hat)
