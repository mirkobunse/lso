 """
    logreg(X, y)

    Objective function of Logistic Regression on data matrix X and label vector y.
"""
logreg(X::Array{Float64,2}, y::Array{Float64,1}, λ::Float64=0.0) = Objective(
    w::Array{Float64,1} -> _logreg_f(X, y, w, λ),  # f
    w::Array{Float64,1} -> _logreg_g(X, y, w, λ),  # g
    f_b = (w::Array{Float64,1}, b::Array{Int,1}) -> _logreg_f(X, y, w, λ, b),
    g_b = (w::Array{Float64,1}, b::Array{Int,1}) -> _logreg_g(X, y, w, λ, b),
    dim = length(y)
)

function _logreg_f(X::Array{Float64,2}, y::Array{Float64,1}, w::Array{Float64,1},
                   λ::Float64, b::Array{Int,1}=Int[])
    reg = 0.0
    if λ > 0.0
        reg = λ * vecnorm(w, 1)
    end
    if length(b) == 0
        return sum( log(1 + exp(-y .* X*w)) )[1] / length(y) + reg
    elseif length(b) == 1   # yes, some machines (or julia versions?) require that
        i = b[1]
        return sum( log(1 + exp(-y[i] .* X[i,:]*w)) )[1] + reg
    else
        return sum( log(1 + exp(-y[b] .* X[b,:]*w)) )[1] / length(b) + reg
    end
end

function _logreg_g(X::Array{Float64,2}, y::Array{Float64,1}, w::Array{Float64,1},
                   λ::Float64, b::Array{Int,1}=Int[])
    reg = 0.0
    if λ > 0.0
        reg = λ * vecnorm(w, 1)
    end
    if length(b) == 0
        return - vec(sum( (y ./ (1 + exp(y .* X*w))) .* X , 1)) ./ length(y) .+ reg
    elseif length(b) == 1   # yes, some machines (or julia versions?) require that
        i = b[1]
        return - vec(sum( (y[i] ./ (1 + exp(y[i] .* X[i,:]*w))) .* X[i,:] , 1)) .+ reg
    else
        return - vec(sum( (y[b] ./ (1 + exp(y[b] .* X[b,:]*w))) .* X[b,:] , 1)) ./ length(b) .+ reg
    end
end

"""
    logreg_predict(w, X)

    Predict the examples in X with Logistic Regression under parameter vector w.
"""
logreg_predict(w::Array{Float64,1}, X::Array{Float64,2}) = sign(vec(1 ./ (1 + exp(-X*w))) -.5)
