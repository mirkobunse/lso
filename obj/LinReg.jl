# 
# Module for objective functions and their gradients
# 
module LinReg


# export f, g, sg, ...


"""
    f(w, X, y)

    Function value of Linear Regression, i.e., vecnorm(y - X*w)^2 / 2
"""
f(w::Array{Float64,1}, X::Array{Float64,2}, y::Array{Float64,1}) = vecnorm(y - X*w)^2 / 2


"""
    g(w, X, y)

    Gradient of Linear Regression, i.e., X' * (X*w - y) / length(y)
"""
g(w::Array{Float64,1}, X::Array{Float64,2}, y::Array{Float64,1}) = X' * (X*w - y) / length(y)


"""
    sg(w, X, y [, i])

    Stochastic gradient of Linear Regression. Will only evaluate the gradient wrt data item
    at index i If i is not provided, a random index is generated.
"""
sg(w::Array{Float64,1}, X::Array{Float64,2}, y::Array{Float64,1}, i::Int32=rand(1:length(y))) = X[i,:]' * (X[i,:]*w - y[i])


end