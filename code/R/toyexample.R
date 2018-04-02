# Construct the likelihood matrix.
e <- 0.5
A <- rbind(c(1,e,e),
           c(e,1,e),
           c(e,e,0.9))

# Fit the mixture model by solving the dual problem using an
# interior-point method (MOSEK).
library(REBayes)
n   <- nrow(A)
k   <- ncol(A)
out <- KWDual(A,rep(1,k),rep(1,n)/n)
x   <- out$f

# Compute the value of the (primal) objective at the MOSEK solution.
cat("Objective value at IP solution:\n")
print(-sum(log(A %*% x)),digits = 12)

# It is easy to see that the MOSEK solution is not optimal:
x0 <- c(0.415,0.415,0.170)
cat("Objective value at better solution:\n")
print(-sum(log(A %*% x0)),digits = 12)

