# TO DO: Explain here briefly what this function does.
function mixGD(L;  w = ones(size(L,2))/size(L,2), maxiter = 10000,
               alpha = 0.1, tol = 1e-6, eps = 1e-8)
    
  # Get the number of rows (n) and columns (k) of the conditional
  # likelihood matrix.
  n = size(L,1);
  k = size(L,2);
    
  # Initialize storage for outputs obj, maxd and timing.
  obj    = zeros(maxiter);
  maxd   = zeros(maxiter);
  timing = zeros(maxiter);
    
  # Repeat until convergence criterion is met, or until the maximum
  # number of iterations is reached.
  iter = 1;
  for iter = 1:maxiter
        
    tic();
        
    # Compute the gradient and objective at the current iterate.
    D         = 1 ./ (L * w + eps);
    obj[iter] = sum(log.(D));
    g         = (L'*D)/n;
        
    # Get the initial search direction.
    p = alpha * g;
        
    # Take initial step along the search direction.
    wnew = take_step(w,p);
        
    # Backtracking line search.
    for i = 1:100
      fnew = 1 ./ (L * wnew + eps);
      if obj[iter] - fnew[1] > n * dot(w - wnew,g)/2
        break;
      else
      p    = p/10;
      wnew = take_step(w,p);
      end
    end

    # Check for convergence.
    maxd[iter] = maximum(abs.(w - wnew));
    timing[iter] = toq();
    if maxd[iter] < tol
      break;
    end
        
    # Move to the new solution estimate.
    w = copy(wnew);
  end

  # Return the mixture weights and other optimization info.
  return w, obj[1:iter], maxd[1:iter], iter, timing[1:iter]
end
    
function take_step(w, g)
  w = proj_simplex(w + g);
  return w
end

function proj_simplex(x)
  u = sort(x, rev = true);
  sv = cumsum(u);
  rho = find(u .> (sv - 1) ./ (1:length(u)))[end]
  theta = max.(0.0, (sv[rho] - 1) ./ rho)
  return max.(x - theta, 0);
end
