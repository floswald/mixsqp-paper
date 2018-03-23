# TO DO:
#  - Add brief description of function and input arguments here.
function mixem(L::Array{Float64,2},
               x::Array{Float64,1} = ones(size(L,2))/size(L,2);
               maxiter::Int = 1000, tol::Float64 = 1e-4,
               eps::Float64 = 1e-15, verbose::Bool = false)

  # Get the number of rows (n) and columns (k) of the likelihood
  # matrix.
  n = size(L,1);
  k = size(L,2);

  # Check input matrix "L". All the entries should be positive.
  if any(L .<= 0)
    throw(ArgumentError("All entries of matrix \"L\" should be positive"))
  end

  # Check input vector "x", and normalize so that the entries sum to 1.
  if any(x .<= 0)
    throw(ArgumentError("All entries of vector \"w\" should be positive"))
  end
  x = x/sum(x);

  # Initialize storage for outputs f (the value of the objective at
  # the current iterate) and d (the maximum difference between each
  # pair of consecutive iterates).
  f = zeros(maxiter);
  d = zeros(maxiter);
  i = 1;
  status = "Did not converge; " *
    @sprintf("reached maximum number of iterations (%d).",maxiter);
    
  # Compute the objective function value at the initial iterate.
  f[i] = -sum(log.(L*x + eps));

  # Preallocate memory for the posterior probabilities and other
  # quantities.
  P  = zeros(n,k);
  x0 = zeros(k);
  z  = zeros(n);
    
  # Print the column labels for reporting the algorithm's progress.
  if verbose
    @printf "iter      objective    delta\n"
  end
    
  # Repeat until convergence criterion is met, or until the maximum
  # number of iterations is reached.
  for i = 2:maxiter

    # Save the current estimate of the solution.
    x0 = x;

    # E STEP
    # ------
    # Compute the posterior probabilities. Note this code is the same
    # as
    #
    #   P = L * diagm(x)
    #   P = P ./ repmat(sum(P,2) + eps,1,k);
    #
    # but substantially more efficient in terms of execution speed and
    # memory allocations. An alternative code that is also fast but
    # perhaps more readable that the code below is this:
    # 
    #   P = x' .* L;
    #   z = sum(P,2) + eps;
    #   P = P ./ z;
    #  
    # Surprisingly, this simple code works because the multiplication
    # and division operations are automatically "broadcasted" across
    # rows or columns.
    broadcast!(*,P,x',L);
    z = sum(P,2) + eps;
    broadcast!(/,P,P,z);
      
    # M STEP
    # ------
    # Update the mixture weights.
    x = mean(P,1)[:];

    # Compute the value of the objective at x.
    f[i] = -sum(log.(L*x + eps));
      
    # Print the status of the algorithm and check the convergence
    # criterion. Convergence is reached when the maximum difference
    # between the mixture weights at two successive iterations is less
    # than the specified tolerance, or when objective increases.
    d[i] = maximum(abs.(x - x0));
    if verbose
      @printf "%4d %0.8e %0.2e\n" i f[i] d[i]
    end
    if d[i] < tol
      status =
        @sprintf("Converged after %d iterations (tolerance = %0.2e).",i,tol);
      break
    end
  end
  if verbose
    @printf "%s" status
    @printf "\n"
  end
    
  # Return the solution (x), the convergence status (status), the
  # objective value at each iteration (f), and the maximum change in
  # the solution at each iteration (d).
  return x, status, f[1:i], d[1:i]
end        
