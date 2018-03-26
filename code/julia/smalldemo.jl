# A small example I am using for testing.

# Setup.
using Distributions
using LowRankApprox
include("datasim.jl");
include("likelihood.jl");
include("mixem.jl");
srand(1);

# Generate a data set with 50,000 samples.
@printf "Generating data set.\n"
x = normtmixdatasim(round(Int,5e4));

# Compute the 50,000 x 20 likelihood matrix.
sd = autoselectmixsd(x,nv = 20);
L  = normlikmatrix(x,sd = sd);

# Fix the mixture model using EM.
@printf "Fitting mixture model using EM.\n"
xem, status, f, d = mixem(L);
println(status);

