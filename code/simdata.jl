# Generate the data sets used in the comparisons of mix-SQP vs. the
# first-order methods (EM and projected gradient).
using Distributions
cinclude("datasim.jl");

m = 2000;

# Generate the matrix.
srand(2019);
z  = normtmixdatasim(round(Int,1e3));
sd = autoselectmixsd(z,nv = m);
L  = normlikmatrix(z,sd = sd);

# Save the matrix to a CSV file.
