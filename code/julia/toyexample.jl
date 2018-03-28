# Load a few packages.
using LowRankApprox
using RCall

# Load some function definitions.
include("misc.jl");
include("mixem.jl");
include("rebayes.jl");
include("mixSQP_old.jl");

e = 0.01;
L = [ 1.0   e   e
      1.0 1.0 1.0
      1.0 1.0 1.0 ];

srand(1);

# Fix the mixture model using EM.
@printf "Fitting mixture model using EM.\n"
@time xem, status, fem, dem = mixem(L);
println(status);

# Fit the mixture model by solving the dual problem using an
# interior-point method.
@printf "Fitting mixture model using REBayes.\n"
@time xreb, freb = mix_rebayes(L);

# Fit the mixture model using the SQP algorithm.
@printf "Fitting mixture model using mixSQP.\n"
@time out = mixSQP(L,lowrank = "nothing");
