# Fit a mixture model using EM. Input argument L is the n x k
# conditional likelihood matrix, where n is the number of samples and
# k is the number of mixture components; optional input argument w is
# the initial estimate of the mixture weights.
mixem <- function (L, w, maxiter = 1e4, tol = 1e-4, eps = 1e-15) {

  # Get the number of mixture components.
  k <- ncol(L)
    
  # Initialize the mixture weights.
  if (missing(w))
    w <- rep(1/k,k)
    
  for (iter in 1:maxiter) {

    # Save the current estimate of the mixture weights.
    w0 <- w

    # E STEP
    # Compute the posterior probabilities
    P <- scale.cols(L,w)
    P <- P / (rowSums(P) + eps)

    # M STEP
    # Update the mixture weights.
    w <- colMeans(P)
    
    # CHECK CONVERGENCE
    # Convergence is reached when the maximum difference between the
    # mixture weights at two successive iterations is less than the
    # specified tolerance.
    if (max(abs(w - w0)) < tol)
      break
  }

  return(w)
}

# Scale each column A[,i] by b[i].
scale.cols <- function (A, b)
  t(t(A) * b)
    
