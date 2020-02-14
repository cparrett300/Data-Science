require(R6)

OLS <- function(y, y_hat) 1/(2*length(y))*sum((y-y_hat)^2)

randn <- function(rows, cols){
  return(matrix(data = rnorm(rows*cols), nrow = rows, ncol = cols))
  
}


MultiLinearRegression <- R6Class("MultiLinearRegression", 
                                                list(
                                                  
                                                  w = NULL,
                                                  b = NULL,
                                                  
                                                  fit = function(X, y, eta = 1e-3, epochs = 1e3, lambda1 =0, show_curve = FALSE) {
                                                    
                                                    N <- nrow(X)
                                                    self$w <- randn(dim(x)[2], dim(y)[2])
                                                    self$b <- randn(1, dim(y)[2])
                                                    
                                                    
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
                                                  predict = function(X)sweep(X %*% self$w, 2, -self$b) #X%*%self$w + self$b
                                                  # x * self.w
                                                ))



