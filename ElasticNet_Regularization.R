require(R6)
LinearRegression_GD_EN_regularization <- R6Class("LinearRegression_GD_EN_regularization", 
                                                list(
                                                  
                                                  w = NULL,
                                                  b = NULL,
                                                                                      #reg_rate is manitude and L1_ratio is ratio of L1 reg     
                                                  fit = function(X, y, eta = 1e-3, epochs = 1e3, reg_rate = 1, L1_ratio = 0.1, show_curve = FALSE) {
                                                    
                                                    N <- nrow(X)
                                                    lambda1 <- reg_rate*L1_ratio
                                                    lambda2 <- reg_rate*(1 - L1_ratio)
                                                    
                                                    
                                                    self$w <- rnorm(ncol(X))
                                                    self$b <- rnorm(1)
                                                    
                                                    J <- vector(mode = "numeric", length = epochs)
                                                    
                                                    for (epoch in seq_len(epochs)){
                                                      y_hat <- self$predict(X)
                                                      J[epoch] <- OLS(y, y_hat) + lambda1/(2*N)*sum(self$w^2) + lambda2/(2*N)*sum(self$w^2)
                                                      self$w <- self$w - eta*(1/N)*(t(X)%*%(y_hat - y) + lambda1*sign(self$w) + lambda2*(self$w)) # %*% matrix multiple
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







