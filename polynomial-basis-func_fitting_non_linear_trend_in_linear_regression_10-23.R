##polynomial basis fun
R2 <- function(y, y_hat)1-sum((y-y_hat)^2)/sum((y- mean(y))^2)

N <- 250

X <- seq(0, 20, length.out = N)

y <- 6.7453 + 4.9373*X + 8.7363*X^2 + rnorm(N, sd= 37)

plot(X, y)



PHI <- cbind(rep(1, N), X, X^2)

lin_reg <- LinearRegression$new()       #initializing the new method
lin_reg$fit(PHI, y)
y_hat <- lin_reg$predict(PHI)

##here we just predicted in PHI instead of X and PhI has X^2 in it

plot(X,y)
lines(X, y_hat)

cat("Training R-squared : ", round(R2(y, y_hat), digits = 4))
