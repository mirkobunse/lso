"""
    logreg(X, y)

    Objective function and gradient of Logistic Regression on data matrix X and label vector y.
"""
logreg(X::Array{Float64,2}, y::Array{Float64,1}) = Objective(
    w::Array{Float64,1} -> sum([ log(1 + exp(-y[i] * X[i,:]*w)) for i=1:length(y) ])[1],
    w::Array{Float64,1} -> sum([ (y[i] / (1 + exp(-y[i] * X[i,:]*w))[1]) * vec(X[i,:]) for i=1:length(y) ], 1)[1]
)