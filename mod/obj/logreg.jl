"""
    logreg(X, y)

    Objective function and gradient of Logistic Regression on data matrix X and label vector y.
"""
logreg(X::Array{Float64,2}, y::Array{Float64,1}) = Objective(
    w::Array{Float64,1} -> sum( log(1 + exp(-y .* X*w)) )[1] / length(y),                   # f
    w::Array{Float64,1} -> - vec(sum( (y ./ (1 + exp(y .* X*w))) .* X , 1)) ./ length(y),   # g
    sf = (w::Array{Float64,1}, i::Array{Int32,1}) -> sum( log(1 + exp(-y[i] .* X[i,:]*w)) )[1] / length(i),
    sg = (w::Array{Float64,1}, i::Array{Int32,1}) -> - vec(sum( (y[i] ./ (1 + exp(y[i] .* X[i,:]*w))) .* X[i,:] , 1)) ./ length(i),
    rng = _rng_sgd(y)
)

logreg_predict(w::Array{Float64,1}, X::Array{Float64,2}) = sign(vec(1 ./ (1 + exp(-X*w))) -.5)