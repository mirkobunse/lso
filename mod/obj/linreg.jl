"""
    linreg(X, y)

    Objective function and gradient of Linear Regression on data matrix X and label vector y.
    linreg(X, y).f(w) = vecnorm(y - X*w)^2 / 2
    linreg(X, y).g(w) = X' * (X*w - y) / length(y)
"""
linreg(X::Array{Float64,2}, y::Array{Float64,1}) = Objective(
    w::Array{Float64,1} -> vecnorm(y - X*w)^2 / (2* length(y)),
    w::Array{Float64,1} -> vec(X' * (X*w - y)) ./ length(y),
    function (w::Array{Float64,1})
        i = rand(1:length(y))
        return X[i,:]' * (X[i,:]*w - y[i])
    end,
    (w::Array{Float64,1}, i::Int32) -> X[i,:]' * (X[i,:]*w - y[i])
)

linreg_predict(w::Array{Float64,1}, X::Array{Float64,2}) = sign(vec(X*w))