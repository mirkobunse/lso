# 
# Module for objective functions and their gradients
# 
module LinReg


# export f, g, sg, ...


"""
    f(w, X, y)

    Function value of Linear Regression, i.e., vecnorm(y - X*w)^2 / 2
"""
f(w, X, y) = vecnorm(y - X*w)^2 / 2


"""
    g(w, X, y)

    Gradient of Linear Regression, i.e., X' * (X*w - y) / length(y)
"""
g(w, X, y) = X' * (X*w - y) / length(y)


"""
    sg(w, X, y [, i])

    Stochastic gradient of Linear Regression. Will only evaluate the gradient wrt data item
    at index i If i is not provided, a random index is generated.
"""
sg(w, X, y, i=rand(1:length(y))) = X[i,:]' * (X[i,:]*w - y[i])


end