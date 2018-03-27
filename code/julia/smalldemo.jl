# A small example I am using for testing.

# Load a few packages.
using Distributions
using LowRankApprox
using RCall

# Load some function definitions.
include("misc.jl");
include("datasim.jl");
include("likelihood.jl");
include("mixem.jl");
include("mixsqp.jl");
include("rebayes.jl");

# Initialize the pseudorandom number generator.
srand(1);

# Generate a data set with 50,000 samples.
@printf "Generating data set.\n"
x = normtmixdatasim(round(Int,5e4));

# Compute the 50,000 x 20 likelihood matrix.
sd = autoselectmixsd(x,nv = 20);
L  = normlikmatrix(x,sd = sd);

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
@time outsqp = mixsqp(L);
