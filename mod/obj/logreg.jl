"""
    logreg(X, y)

    Objective function of Logistic Regression on data matrix X and label vector y.
"""
logreg(X::Array{Float64,2}, y::Array{Float64,1}) = Objective(
    w::Array{Float64,1} -> _logreg_f(X, y, w),  # f
    w::Array{Float64,1} -> _logreg_g(X, y, w),  # g
    f_b = (w::Array{Float64,1}, b::Array{Int32,1}) -> _logreg_f(X, y, w, b),
    g_b = (w::Array{Float64,1}, b::Array{Int32,1}) -> _logreg_g(X, y, w, b),
    dim = length(y)
)

function _logreg_f(X::Array{Float64,2}, y::Array{Float64,1}, w::Array{Float64,1},
                   b::Array{Int32,1}=Int32[])
    if length(b) == 0
        return sum( log(1 + exp(-y .* X*w)) )[1] / length(y)
    else
        return sum( log(1 + exp(-y[b] .* X[b,:]*w)) )[1] / length(b)
    end
end

function _logreg_g(X::Array{Float64,2}, y::Array{Float64,1}, w::Array{Float64,1},
                   b::Array{Int32,1}=Int32[])
    if length(b) == 0
        return - vec(sum( (y ./ (1 + exp(y .* X*w))) .* X , 1)) ./ length(y)
    else
        return - vec(sum( (y[b] ./ (1 + exp(y[b] .* X[b,:]*w))) .* X[b,:] , 1)) ./ length(b)
    end
end

"""
    logreg_predict(w, X)

    Predict the examples in X with Logistic Regression under parameter vector w.
"""
logreg_predict(w::Array{Float64,1}, X::Array{Float64,2}) = sign(vec(1 ./ (1 + exp(-X*w))) -.5)