require(R6)

LinearRegression <- R6Class("LinearRegression",
  list(
    
    w = NULL,
    
    fit = function(X, y) self$w <-solve(t(X)%*%X, t(X)%*%y),
    
    predict = function(X) X%*%self$w
    
  )
)   

# name should be same as class definition name