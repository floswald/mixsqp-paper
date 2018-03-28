# Reconstruct the matrix from the partial QR decomposition.
function reconstructmatrixqr(F::LowRankApprox.PartialQR{Float64},
                             P::SparseMatrixCSC{Float64,Int})
  return F[:Q] * (F[:R] * P')
end

# Reconstruct the matrix from the partial SVD factorization.
function reconstructmatrixsvd(F::LowRankApprox.PartialSVD{Float64,Float64},
                              S::Diagonal{Float64})
  return F[:U] * (S * F[:Vt])
end

# Compute the gradient and Hessian of the (primal) objective.
function computegradient(L::Array{Float64,2}, x::Array{Float64,1},
                         eps::Float64)
  n = nrow(L);
  k = ncol(L);
  d = 1./(L*x + eps);
  g = -L'*d/n;
  H = L'*Diagonal(d.^2)*L/n + eps*eye(k);    
  return g, H;
end

# The same as computegradient, but faster, and with more efficient use
# of memory allocation. Note that it is important to avoid taking the
# transpose of the (potentially large) matrix L; for example, for
# computing the gradient, the equivalent code g = -L'*d/n is slower
# and requires more memory allocations.
function computegradient!(L::Array{Float64,2}, x::Array{Float64,1},
                          g::Array{Float64,1}, H::Array{Float64,2},
                          d::Array{Float64,1}, Ld::Array{Float64,2},
                          eps::Float64)
  n    = nrow(L);
  k    = ncol(L);
  d[:] = 1./(L*x + eps);
  g[:] = -(d'*L)'/n;
  d[:] = d.^2;
  transpose!(Ld,L);
  broadcast!(*,Ld,d',Ld);
  H[:] = (Ld*L)/n + eps*eye(k);
  return 0
end

# TO DO: Explain here what this function does, and how to use it.
# ind = working set
function solveqp (ind::Array{Int,1}, g::Array{Float64,1}, H::Array{Float64,2},
                  maxiter::Int, tol::Float64)

  # Set the initial guess.
  k      = length(g);
  x      = zeros(k);
  x[ind] = 1/k;

  # Repeat until the convergence criterion is met, or until we reach
  # the maximum number of iterations.
  for i = 1:maxiter

    # Define the quadratic program.
    s  = length(ind);
    Hs = H[ind,ind];
    d  = H*x + 2*g + 1;
    ds = d[ind];

    # Compute the search direction.
    ps     = -Hs \ ds;
    p      = zeros(k);
    p[ind] = ps;

    # Check convergence using the KKT conditions.
    if norm(p_s) < tol
            
      # Compute the Lagrange multiplier.
      lambda = d - minimum(ds);
      if all(lambda .>= 0)
        break
      else
          
        # Add to the active set the entry corresponding to the
        # smallest (or largest?) Lagrange multiplier.
        r, j = findmin(lambda);
        ind  = sort([ind; j]);
      end
    else

      # Move to the new "inner loop" iterate (y) along the search
      # direction.
##         y = y + alpha * p;
        end
  end
end

##         # Find a feasible step length.
##         alpha     = 1;
##         alpha0    = -y[ind]./p_s;
##         ind_block = find(p_s .< 0);
##         alpha0    = alpha0[ind_block];
##         if ~isempty(ind_block)
##           v, t = findmin(alpha0);
##           if v < 1

##             # Blocking constraint.
##             ind_block = ind[ind_block[t]]; 
##             alpha     = v;
              
##             # Update working set if there is a blocking constraint.
##             deleteat!(ind,find(ind - ind_block .== 0));
##           end
##         end
          
##       end
##     end


# This function implements the main loop of mixsqp.
function mixsqploop!(L::Array{Float64,2}, x::Array{Float64,1},
                     maxiter::Int, maxqpiter::Int, convtol::Float64,
                     sptol::Float64, eps::Float64, verbose::Bool)

  # Get the number of rows (n) and columns (k) of the likelihood
  # matrix.
  n = nrow(L);
  k = ncol(L);
    
  # Initialize loop variables used in the loop below so that they are
  # available outside the scope of the loop.
  i = 0;

  # Preallocate memory for additional quantities computed inside the
  # loop.
  g  = zeros(k);
  d  = zeros(n);
  H  = zeros(k,k);
  Ld = zeros(k,n);
    
  # Repeat until we reach the maximum number if iterations, or until
  # the convergence criterion is met.
  for i = 1:maxiter

    # COMPUTE GRADIENT AND HESSIAN
    # This is equivalent to the following code, but faster:
    #
    #   g, H = computegradient(L,x,eps)
    #
    computegradient!(L,x,g,H,d,I,Ld,eps);

    # CHECK CONVERGENCE
    # Check convergence of outer loop.
    if minimum(g + 1) >= -convtol
      break
    end
      
      
  end

  return x
end

# TO DO: Add comments here explaining what this function does, and
# what are the inputs and outputs.
function mixsqp(L::Array{Float64,2},
                x::Array{Float64,1} = ones(ncol(L))/ncol(L);
                lowrankapprox = "svd", maxiter::Int = 1000,
                maxqpiter::Int = 100, convtol::Float64 = 1e-8,
                sptol::Float64 = 1e-6, factol::Float64 = 1e-15,
                eps::Float64 = 1e-15, verbose::Bool = true)
    
  # Get the number of rows (n) and columns (k) of the likelihood
  # matrix.
  n = nrow(L);
  k = ncol(L);

  # Check input matrix "L". All the entries should be positive.
  if any(L .<= 0)
    throw(ArgumentError("All entries of matrix \"L\" should be positive"));
  end

  # Check input vector "x", and normalize so that the entries sum to 1.
  if any(x .<= 0)
    throw(ArgumentError("All entries of vector \"w\" should be positive"));
  end
  if (length(x) != k)
    throw(ArgumentError("Input vector \"x\" should have one entry for " *
                        "each column of L"));
  end
  x = x/sum(x);

  # Check input argument "lowrankapprox".
  if (!(lowrankapprox == "qr" ||
        lowrankapprox == "svd" ||
        lowrankapprox == "none"))
    throw(ArgumentError("Argument \"lowrankapprox\" should be \"qr\", " *
                        "\"svd\" or \"none\""));
  end
    
  # Summarize the analysis here.
  if verbose
    @printf "Running SQP algorithm with the following settings:\n"
    @printf " - %d x %d data matrix\n" n k
    if lowrankapprox == "qr"
      @printf " - Using SVD approximation with "
      @printf "%0.2e error tolerance\n" factol
    elseif lowrankapprox == "svd"
      @printf " - Using QR approximation with "
      @printf "%0.2e error tolerance\n" factol
    end
    @printf " - maximum number of iterations = %d\n" maxiter
  end
    
  # COMPUTE (PARTIAL) FACTORIZATION OF LIKELIHOOD MATRIX
  # ----------------------------------------------------
  # If requested, compute a partial QR or partial SVD factorization of
  # the likelihood matrix using the LowRankApprox package. For
  # details, see https://github.com/klho/LowRankApprox.jl.
  out, fac_elapsed, fac_bytes, gctime,
  memallocs = @timed if lowrankapprox == "qr"
    if verbose
      @printf "Computing partial QR factorization.\n"
    end
    F = pqrfact(L,rtol = factol);
    P = convert(SparseMatrixCSC{Float64,Int},F[:P]);
  elseif lowrankapprox == "svd"
    if verbose
      @printf "Computing partial SVD factorization.\n"
    end
    F = psvdfact(L,rtol = factol);
    S = Diagonal(F[:S]);
  end

  # Report accuracy and computational expense of factorization.
  if verbose
    @printf(" - Factorization took %0.4f seconds (allocation: %0.2f MiB)\n",
            fac_elapsed,fac_bytes/1024^2);
    if lowrankapprox != "none"
      if lowrankapprox == "qr"
        @printf(" - Min. value in partial QR approx. = %0.2e\n",
                minimum(reconstructmatrixqr(F,P)));
        @printf(" - Max. error in partial QR approx. = %0.2e\n",
                maximum(abs.(reconstructmatrixqr(F,P) - L)));
      elseif lowrankapprox == "svd"
        @printf(" - Min. value in partial SVD approx. = %0.2e\n",
                minimum(reconstructmatrixsvd(F,S)));
        @printf(" - Max. error in partial SVD approx. = %0.2e\n",
                maximum(abs.(reconstructmatrixsvd(F,S) - L)));
      end
    end
  end

  # RUN SQP ALGORITHM
  # -----------------
  if verbose
    @printf "Running SQP algorithm.\n"
  end
  x, loop_elapsed, loop_bytes, gctime,
  memallocs = @timed mixsqploop!(L,x,maxiter,maxqpiter,convtol,sptol,
                                 eps,verbose);
  @printf(" - Optimization took %0.4f seconds (allocation: %0.2f MiB)\n",
            loop_elapsed,loop_bytes/1024^2);

  return x
end

##   # Initialize storage for the outputs obj, gmin, nnz and nqp.
##   obj    = zeros(maxiter);
##   gmin   = zeros(maxiter);
##   nnz    = zeros(maxiter);
##   nqp    = zeros(maxiter);
##   timing = zeros(maxiter);
    
##   j = 0;
##   D = 0;

##   # Print the column labels for reporting the algorithm's progress.
##   if verbose
##     @printf("iter       objective -min(g+1) #nnz #qp\n")
##   end

##   # QP subproblem start.
##   for i = 1:maxiter

##     # Compute the gradient and Hessian, optionally using the partial
##     # QR decomposition to increase the speed of these computations.
##     # gradient and Hessian computation -- Rank reduction method
##     if lowrank == "qr"
##         D = 1./(F[:Q]*(F[:R]*(P'*x)) + eps);
##         g = -P * F[:R]' * (F[:Q]'*D)/n;
##         H = P * F[:R]' * (F[:Q]'*Diagonal(D.^2)*F[:Q]) * F[:R] * P'/n + eps * eye(k);
##     elseif lowrank == "svd"
##         D = 1./(F[:U]*(S*(F[:Vt]*x)) + eps);
##         g = -F[:Vt]'*(S * (F[:U]'*D))/n;
##         H = (F[:V]*S*(F[:U]'*Diagonal(D.^2)*F[:U])* S*F[:Vt])/n + eps * eye(k);
##     end

##     # Report on the algorithm's progress.
##     #
##     # TO DO: The L * x matrix operation here used to compute the
##     # objective function could dramatically slow down the algorithm
##     # when number of QR factors in the partial QR is much smaller than
##     # k. We need to think of a way to avoid this by having an option
##     # to not output the objective function at each iteration, and/or
##     # make sure that this objective function operation is not included
##     # in the timing.
##     #
##     obj[i]  = -sum(log.(L * x + eps));
##     gmin[i] = minimum(g + 1);
##     nnz[i]  = sum(x .> sptol);
##     nqp[i]  = j;
##     if verbose
##       @printf("%4d %0.8e %+0.2e %4d %3d\n",i,obj[i],-gmin[i],nnz[i],j);
##     end
      

##     # Update the solution to the original optimization problem.
##     x = y;

##   end

##   # Return: (1) the solution (after zeroing out any values below the
##   # tolerance); (2) the value of the objective at each iteration; (3)
##   # the minimum gradient value of the modified objective at each
##   # iteration; (4) the number of nonzero entries in the vector at each
##   # iteration; and (5) the number of inner iterations taken to solve
##   # the QP subproblem at each outer iteration.
##   x[x .< sptol] = 0; x = x/sum(x);
##   totaltime = lowranktime + sum(timing[1:i]);
##   if verbose
##     @printf("Optimization took %d iterations and %0.4f seconds.\n",i,totaltime)
##   end

##   return Dict([("x",full(x)), ("totaltime",totaltime), ("lowranktime",lowranktime),
##                ("obj",obj[1:i]), ("gmin",gmin[1:i]), ("nnz",nnz[1:i]),
##                ("nqp",nqp[1:i]), ("timing",timing[1:i])])
## end
