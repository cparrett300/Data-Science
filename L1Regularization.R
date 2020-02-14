require(R6)

OLS <- function(y, y_hat) 1/(2*length(y))*sum((y-y_hat)^2)

LinearRegression_GD_L1regularization <- R6Class("LinearRegression_GD_L1regularization", 
                                              list(
                                                
                                                w = NULL,
                                                b = NULL,
                                                
                                                fit = function(X, y, eta = 1e-3, epochs = 1e3, lambda1 =0, show_curve = FALSE) {
                                                  
                                                  N <- nrow(X)
                                                  
                                                  self$w <- rnorm(ncol(X))
                                                  self$b <- rnorm(1)
                                                  
                                                  J <- vector(mode = "numeric", length = epochs)
                                                  
                                                  for (epoch in seq_len(epochs)){
                                                    y_hat <- self$predict(X)
                                                    J[epoch] <- OLS(y, y_hat) + lambda1/(2*N)*sum(self$w^2)
                                                    self$w <- self$w - eta*(1/N)*(t(X)%*%(y_hat - y) + lambda1*sign(self$w)) # %*% matrix multiple
                                                    self$b <- self$b - eta*(1/N)*sum(y_hat-y)
                                                  }
                                                  
                                                  if (show_curve){
                                                    plot(seq_len(epochs), J, type = "l", main = "Training Curve", xlab = "epochs", ylab = "J")
                                                  }
                                                  
                                                },
                                                # w = (x.T * x)^-1 * x.T * y
                                                predict = function(X) X%*%self$w + self$b
                                                # x * self.w
                                              ))







N <- 50

X <- seq(0,20, length.out = N)

y <- 5.938 + 3.1415672*X + rnorm(N, sd = 1.7)

y[(N-10):(N-7)] <- y[(N-10):(N-7)] + 30

plot(X,y)

PHI <- cbind(rep(1,N), X)
X <- matrix(X)

lin_reg <- LinearRegression_GD_L1regularization$new()

lin_reg$fit(X,y,lambda1 = 200)

y_hat <- lin_reg$predict(X)

plot(X,y)
lines(X,y_hat)
