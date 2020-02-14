#this s not generalizable enough 
##Linear Regression

N <- 100

x <- seq(0, 20, length.out = N) ## N evenly spaced number between 0 to 20
y <- 5.5242 + 3.0928*x + rnorm(N, sd = 2.5)  #rnorm generates bunch of random values

#plot(x, y)

d <- mean(x^2) - mean(x)^2    #denominator

w0 <- (mean(x^2)*mean(y) - mean(x)*mean(x*y))/d
w1 <- mean(mean(x*y) - mean(x)*mean(y))/d

y_hat <- w0+w1*x

plot(x, y)
lines(x, y_hat)

#r squared
R2 <- 1- sum((y-y_hat)^2)/sum((y-mean(y))^2)

cat("Training R-squared: ", round(R2, digits = 3))


