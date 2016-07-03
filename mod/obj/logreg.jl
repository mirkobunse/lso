
f_logreg(w::Array{Float64,1}, X::Array{Float64,2}, y::Array{Float64,1}) = sum([ log(1 + exp(-y[i] * X[i,:]*w)) for i=1:length(y) ])[1]

g_logreg(w::Array{Float64,1}, X::Array{Float64,2}, y::Array{Float64,1}) = sum([ (y[i] / (1 + exp(-y[i] * X[i,:]*w))[1]) * vec(X[i,:]) for i=1:length(y) ], 1)[1]
