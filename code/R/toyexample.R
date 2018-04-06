# NOTE: This uses a modified version of the REBayes package in which
# KWDual also outputs the MOSEK problem specification (P).
library(REBayes)
library(Rmosek)
source("mixem.R")

# FIRST EXAMPLE
# -------------
# Construct the likelihood matrix.
cat("FIRST EXAMPLE\n")
n <- 3
k <- 3
e <- 0.5
A <- rbind(c(1,e,e),
           c(e,e,0.9),
           c(e,1,e))

# Fit the mixture model using EM.
xem <- mixem(A,tol = 1e-8)

# Fit the mixture model by solving the dual problem using an
# interior-point method (MOSEK).
out <- KWDual(A,rep(1,k),rep(1,n)/n)
x   <- out$f
P1  <- out$P

# Write the MOSEK problem specification to file.
r <- mosek_write(P1,"P1.mps",opts = list(scofile = "P1.sco",verbose = 0))

# In this first example, the MOSEK and EM solutions are equally good.
cat(sprintf("Objective value at IP solution: %0.8f\n",-sum(log(A %*% x))))
cat(sprintf("Objective value at EM solution: %0.8f\n",-sum(log(A %*% xem))))

# SECOND EXAMPLE
# --------------
# Construct the likelihood matrix.
cat("SECOND EXAMPLE\n")
e <- 0.5
A <- rbind(c(1,e,e),
           c(e,1,e),
           c(e,e,0.9))

# Fit the mixture model using EM.
xem <- mixem(A,tol = 1e-8)

# Fit the mixture model by solving the dual problem using an
# interior-point method (MOSEK).
out <- KWDual(A,rep(1,k),rep(1,n)/n)
x   <- out$f
P2  <- out$P

# Write the MOSEK problem specification to file.
r <- mosek_write(P2,"P2.mps",opts = list(scofile = "P2.sco",verbose = 0))

# In this second example, it is easy to see that the MOSEK solution is
# not optimal when we compare against the value of the objective at the
# EM solution.
cat(sprintf("Objective value at IP solution: %0.8f\n",-sum(log(A %*% x))))
cat(sprintf("Objective value at EM solution: %0.8f\n",-sum(log(A %*% xem))))
