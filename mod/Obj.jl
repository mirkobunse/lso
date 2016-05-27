# 
# Module for objective functions and their gradients
# 
module Obj


# export lr_f, lr_g, lr_sg, ...


"""
    lr_f(w, X, y)

    Function value of Linear Regression, i.e., vecnorm(y - X*w)^2 / 2
"""
lr_f(w, X, y) = vecnorm(y - X*w)^2 / 2


"""
    lr_g(w, X, y)

    Gradient of Linear Regression, i.e., X' * (X*w - y) / length(y)
"""
lr_g(w, X, y) = X' * (X*w - y) / length(y)


"""
    lr_sg(w, X, y [, i])

    Stochastic gradient of Linear Regression. Will only evaluate the gradient wrt data item
    at index i If i is not provided, a random index is generated.
"""
lr_sg(w, X, y, i=rand(1:length(y))) = X[i,:]' * (X[i,:]*w - y[i])


end