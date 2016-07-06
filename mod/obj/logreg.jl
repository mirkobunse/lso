"""
    logreg(X, y)

    Objective function and gradient of Logistic Regression on data matrix X and label vector y.
"""
logreg(X::Array{Float64,2}, y::Array{Float64,1}) = Objective(
    w::Array{Float64,1} -> sum( log(1 + exp(-y .* X*w)) )[1],
    function (w::Array{Float64,1})
        - vec(sum( (y ./ (1 + exp(y .* X*w))) .* X , 1))
    end
)

logreg_predict(w::Array{Float64,1}, X::Array{Float64,2}) = sign(vec(1 ./ (1 + exp(-X*w))) -.5)