# TO DO: Explain here what this function does, and how to use it.
function REBayes(L::Array{Float64,2}, eps::Float64 = 1e-15)

  # Check input matrix "L". All the entries should be positive.
  if any(L .<= 0)
    throw(ArgumentError("All entries of matrix \"L\" should be positive"))
  end
    
  # Copy the likelihood matrix to the R environment.
  @rput L;

  # Solve the dual optimization problem using MOSEK.
  R"""
  n <- nrow(L)
  k <- ncol(L)
  x <- REBayes::KWDual(L,rep(1,k),rep(1,n)/n)$f
  """

  # Make sure all the entries are positive, and re-normalized as needed.
  @rget x;
  x[x .<  0] = 0;
  x = x/sum(x); 
    
  # Return the REBayes solution, and the value of the objective at the
  # solution.
  return x, mixobjective(L,x,eps)
end
