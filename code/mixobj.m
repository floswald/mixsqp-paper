% Compute the objective, and gradient of this objective, optimized by
% mix-SQP.
function [f, g] = mixobj (L, x, e)
  m = numel(x);
  y = L*x + e;
  if any(y <= 0)
    f = Inf;
    g = zeros(m,1);  
  else
    n = size(L,1);
    f = -sum(log(y));
    d = 1./(y + e);
    g = -(d'*L)';
  end
