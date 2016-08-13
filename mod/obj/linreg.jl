"""
    linreg(X, y)

    Objective function and gradient of Linear Regression on data matrix X and label vector y.
    linreg(X, y).f(w) = vecnorm(y - X*w)^2 / 2
    linreg(X, y).g(w) = X' * (X*w - y) / length(y)
"""
linreg(X::Array{Float64,2}, y::Array{Float64,1}) = Objective(
    w::Array{Float64,1} -> vecnorm(y - X*w)^2 / (2*length(y)), # f
    w::Array{Float64,1} -> vec(X' * (X*w - y)) ./ length(y),    # g
    f_b = (w::Array{Float64,1}, i::Array{Int32,1}) -> vecnorm(y[i] - X[i,:]*w)^2 / (2*length(i)),
    g_b = (w::Array{Float64,1}, i::Array{Int32,1}) -> X[i,:]' * (X[i,:]*w - y[i]) ./ length(i),
    dim = length(y)
)

linreg_predict(w::Array{Float64,1}, X::Array{Float64,2}) = sign(vec(X*w))