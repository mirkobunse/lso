"""
    linreg(X, y)

    Objective function of Linear Regression on data matrix X and label vector y.
"""
linreg(X::Array{Float64,2}, y::Array{Float64,1}) = Objective(
    w::Array{Float64,1} -> _linreg_f(X, y, w),  # f
    w::Array{Float64,1} -> _linreg_g(X, y, w),  # g
    f_b = (w::Array{Float64,1}, b::Array{Int,1}) -> _linreg_f(X, y, w, b),
    g_b = (w::Array{Float64,1}, b::Array{Int,1}) -> _linreg_g(X, y, w, b),
    dim = length(y)
)

function _linreg_f(X::Array{Float64,2}, y::Array{Float64,1}, w::Array{Float64,1},
                   b::Array{Int,1}=Int[])
    if length(b) == 0
        return vecnorm(y - X*w)^2 / (2*length(y))
    elseif length(b) == 1   # yes, some machines (or julia versions?) require that
        i = b[1]
        vecnorm(y[i] - X[i,:]*w)^2 / 2
    else
        return vecnorm(y[b] - X[b,:]*w)^2 / (2*length(b))
    end
end

function _linreg_g(X::Array{Float64,2}, y::Array{Float64,1}, w::Array{Float64,1},
                   b::Array{Int,1}=Int[])
    if length(b) == 0
        return vec(X' * (X*w - y)) ./ length(y)
    elseif length(b) == 1   # yes, some machines (or julia versions?) require that
        i = b[1]
        return vec(X[i,:]' * (X[i,:]*w - y[i]))
    else
        return vec(X[b,:]' * (X[b,:]*w - y[b])) ./ length(b)
    end
end

"""
    linreg_predict(w, X)

    Predict the examples in X with Linear Regression under parameter vector w.
"""
linreg_predict(w::Array{Float64,1}, X::Array{Float64,2}) = sign(vec(X*w))
