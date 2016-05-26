# 
# Module for objective functions and their gradients
# 
module Obj
	
export f_linreg

"""
    f_linreg(w, X, y)

	Function value of Linear Regression
"""
function f_linreg(w, X, y) #(w::Array{Float32,1}, X::Array{Float32,2}, y::Array{Float32,1})
	return vecnorm(y - X*w)^2 / 2
end
	
end