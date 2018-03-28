# Fit a mixture model by solving the dual optimization problem using
# the MOSEK interior-point solver. Input argument L is the n x k
# likelihood matrix, where n is the number of samples and k is the
# number of mixture components.
function mix_rebayes(L::Array{Float64,2}, eps::Float64 = 1e-15)

  # Check input matrix "L". All the entries should be positive, and it
  # should have at least 2 columns.
  if (k < 2)
    throw(ArgumentError("Argument \"L\" should have at least 2 columns"));
  end
  if any(L .<= 0)
    throw(ArgumentError("All entries of matrix \"L\" should be positive"));
  end
    
  # Copy the likelihood matrix to the R environment.
  @rput L;

  # Solve the dual optimization problem using MOSEK.
  R"""
  n <- nrow(L)
  k <- ncol(L)
  x <- REBayes::KWDual(L,rep(1,k),rep(1,n)/n)$f
  """

  # Make sure all the entries are positive, and re-normalize as needed.
  @rget x;
  x[x .< 0] = 0;
  x = x/sum(x); 
    
  # Return the REBayes solution, and the value of the objective at the
  # solution.
  f = mixobjective(L,x,eps);
  return x, f
end
