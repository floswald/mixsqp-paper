# Return the number of rows of matrix A.
function nrow(A::Array{Float64,2})
  return size(A,1)
end

# Return the number of columns of matrix A.
function ncol(A::Array{Float64,2})
  return size(A,2)
end

# Compute the value of the (primal) objective at x.
function mixobjective(L::Array{Float64,2}, x::Array{Float64,1}, eps::Float64)
  return -sum(log.(L*x + eps))
end
