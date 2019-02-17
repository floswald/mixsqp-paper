# This is the script used to generate the results comparing mix-SQP
# vs. the EM algorithm. The results for the projected gradient method
# are generated separately in run_pg.m after first running this
# script. To run on the RCC cluster, I set up my environment with the
# following commands:
#
# sinteractive --partition=broadwl --mem=8G
# module load julia/0.6.2
#
# Note that I also ran this script with m = 200 and m = 2000.
n = 20000;
m = 20;
matrixfile = "simdata-n=20000-m=20.csv";

using Distributions
using LowRankApprox
include("datasim.jl");
include("likelihood.jl");
include("mixEM.jl");
include("mixGD.jl");
include("mixSQP.jl");

# Generate the matrix.
srand(2019);
@printf "Generating %d x %d data matrix.\n" n m
z  = normtmixdatasim(n);
sd = autoselectmixsd(z,nv = m);
L  = normlikmatrix(z,sd = sd);

# Save the matrix to a CSV file.
@printf "Writing data to CSV file.\n"
writecsv(matrixfile,L);

# Run the methods for a small number of iterations to precompile the
# code.
@printf "Precompiling EM and mixSQP code.\n"
xem, fem, tem = mixEM(L,maxiter = 10);
outsqp1 = mixSQP(L,lowrank = "none",eps = 1e-8,maxqpiter = m,
                 maxiter = 20,verbose = false);
outsqp2 = mixSQP(L,lowrank = "qr", eps = 1e-8, maxqpiter = m,
                 maxiter = 200,verbose = false);

# Run the EM algorithm.
@printf "Fitting model using EM.\n"
@time xem, fem, tem = mixEM(L,maxiter = 10000,tol = 1e-6);

# Run mix-SQP with no approximation to the input matrix.
@printf "Fitting model using mix-SQP with exact L.\n"
outsqp1 = mixSQP(L,lowrank = "none",eps = 1e-8,maxqpiter = m,
                 maxiter = 200,verbose = false);

# Run the mix-SQP with a low rank (truncated QR) approximation to the
# input matrix.
@printf "Fitting model using mix-SQP with approximate L.\n"
outsqp2 = mixSQP(L,lowrank = "qr", eps = 1e-8, maxqpiter = m,
                 maxiter = 200,verbose = false);

# Compare the quality of the solutions.
@printf "Objective at SQP1 solution: %0.12e\n" mixobjective(L,outsqp1["x"])
@printf "Objective at SQP2 solution: %0.12e\n" mixobjective(L,outsqp2["x"])
@printf "Objective at EM   solution: %0.12e\n" mixobjective(L,xem)
