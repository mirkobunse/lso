module Obj


import LsoBase


"""
    Objective(f, g [, sg, h])

    Creates an objective function object with:
     f:     w  ->   f(w)
     g:     w  -> ∇ f(w)
    sg: (w, i) -> ∇ f(w) (stochastic with index i)
     h:     w ->  ∇²f(w)

    If sg is not provided, g is used for sg.
    If h is not provided, an identity matrix is used.
"""
type Objective
    f::Function  # w ->   f(w)
    g::Function  # w -> ∇ f(w)
    sg::Function # (w, i) -> ∇f(w) (stochastic with index i)
    h::Function  # w -> ∇²f(w)
    function Objective(f::Function, g::Function,
                       sg::Function = g,
                       h::Function = (w -> eye(length(w))))
        new(f, g, sg, h)
    end
end


"""
    Random index generator for stochastic gradient.
"""
sgr(y) = rand(1:length(y))



# include ./obj/*
LsoBase.includedir(dirname(@__FILE__)*"/obj")



end