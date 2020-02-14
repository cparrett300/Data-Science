source("C:/Users/Aashish/techfield/algorithms/multilinear_regression/Multi_Linear_Regression.R")
N <- 1000

X <- seq(0,20, length.out = N)
X <- matrix(X)
y1 <- 5.938 + 3.1415672*X + rnorm(N, sd = 10)
y2 <- 8.54656 - 5.3423*X + rnorm(N, sd = 10)



y1[(N-100):(N-70)] <- y1[(N-100):(N-70)] + 25
y2[(N-50):(N-100)] <- y2[(N-50):(N-100)] - 20

y <- cbind(y1, y2)

plot(X,y1)
plot(X,y2)


lin_reg <- MultiLinearRegression$new()

lin_reg$fit(X,y)

y_hat <- lin_reg$predict(X)


y_hat


plot(X,y1)
lines(X,y_hat[,1])
plot(X,y2)
lines(X,y_hat[,2])




