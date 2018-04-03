library(REBayes)
source("mixem.R")

# Construct the likelihood matrix.
e <- 0.5
A <- rbind(c(1,e,e),
           c(e,1,e),
           c(e,e,0.9))

# Fit the mixture model using EM.
xem <- mixem(A,tol = 1e-8)

# Fit the mixture model by solving the dual problem using an
# interior-point method (MOSEK).
n   <- nrow(A)
k   <- ncol(A)
out <- KWDual(A,rep(1,k),rep(1,n)/n)
x   <- out$f

# Compute the value of the (primal) objective at the MOSEK solution.
cat("Objective value at IP solution:\n")
print(-sum(log(A %*% x)),digits = 12)

# It is easy to see that the MOSEK solution is not optimal when we
# compare to the quality of the EM solution:
cat("Objective value at EM solution:\n")
print(-sum(log(A %*% xem)),digits = 12)

