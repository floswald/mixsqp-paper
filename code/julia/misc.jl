# Compute the value of the (primal) objective at x.
function mixobjective(L::Array{Float64,2}, x::Array{Float64,1}, eps::Float64)
  return -sum(log.(L*x + eps))
end
