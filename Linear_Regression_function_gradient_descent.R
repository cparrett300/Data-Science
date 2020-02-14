require(R6)

OLS <- function(y, y_hat) 1/(2*length(y))*sum((y-y_hat)^2)



LinearRegression <- R6Class("LinearRegression",
                            list(
                              
                              w = NULL,
                              
                              fit = function(X, y, eta = 1e-3, epochs = 1e3, show_curve = FALSE) {
                                
                                self$w <-rnorm(ncol(X))
                                
                                J <- vector(mode = "numeric", length = epochs)
                                
                                for(epoch in seq_len(epochs)) {
                                  y_hat <- self$predict(X)
                                  J[epoch] <- OLS(y, y_hat)
                                  self$w <- self$w - eta*(1/length(X))*t(X)%*%(y_hat - y)
                                }
                                
                                if(show_curve){
                                  plot(seq_len(epochs), J, type = "l", main = "Training Curve", xlab = "epochs", ylab="J")
                                }
                                
                              },
                              predict = function(X) X%*%self$w
                              
                            )
)   

