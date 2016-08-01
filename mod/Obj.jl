module Obj


import LsoBase


"""
    Objective(f, g [, sg, sgi, h])

    Creates an objective function object with:
      f:     w  ->   f(w)
      g:     w  -> ∇ f(w)
      h:     w  -> ∇²f(w)
     sf: (w, i) ->   f(w) (stochastic with index i)
     sg: (w, i) -> ∇ f(w) (stochastic with index i)
    rng:     () ->     i  (random index generator)

    If sf is not provided, f is used for sf.
    If sg is not provided, g is used for sg.
    If h is not provided, an identity matrix is used.
"""
type Objective
      f::Function
      g::Function
      h::Function
     sf::Function
     sg::Function
    rng::Function
    function Objective(f::Function, g::Function;
                       h::Function = (w -> eye(length(w))),
                       sf::Function = (w, i) -> f(w),
                       sg::Function = (w, i) -> g(w),
                       rng::Function = () -> -1)
        new(f, g, h, sf, sg, rng)
    end
end


"""
    Returns a random index generator function for stochastic gradient.
"""
_rng_sgd(y::Array{Float64,1}) = batchSize::Int32 -> randperm(length(y))[1:batchSize]



# include ./obj/*
LsoBase.includedir(dirname(@__FILE__)*"/obj")



end