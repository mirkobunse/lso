# 
# Module for objective functions and their gradients
# 
module Obj


# export f_lr, g_lr, sg_lr, ...


"""
    f_lr(w, X, y)

    Function value of Linear Regression, i.e., vecnorm(y - X*w)^2 / 2
"""
f_lr(w, X, y) = vecnorm(y - X*w)^2 / 2


"""
    g_lr(w, X, y)

    Gradient of Linear Regression, i.e., X' * (X*w - y) / length(y)
"""
g_lr(w, X, y) = X' * (X*w - y) / length(y)


"""
    sg_lr(w, X, y [, i])

    Stochastic gradient of Linear Regression. Will only evaluate the gradient wrt data item
    at index i If i is not provided, a random index is generated.
"""
sg_lr(w, X, y, i=rand(1:length(y))) = X[i,:]' * (X[i,:]*w - y[i])


end