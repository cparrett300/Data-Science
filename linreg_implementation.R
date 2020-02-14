R2 <- function(y, y_hat) 1-sum((y - y_hat)^2)/ sum((y - mean(y))^2)

N <- 250

X <- seq(0, 20, length.out = N)

y <- 4.39387 + 2.385*X + rnorm(N, sd = 2)


X <- cbind(rep(1, N), X)

lin_reg <- LinearRegression$new()       #initializing the new method
lin_reg$fit(X, y)
y_hat <- lin_reg$predict(X)


plot(X[,2],y)
lines(X[,2], y_hat)

cat("Training R-squared : ", round(R2(y, y_hat), digits = 3))

print(lin_reg$w)

save(lin_reg, file = "Lin_reg.RData")
  