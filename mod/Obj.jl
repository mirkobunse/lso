module Obj


import LsoBase


"""
    Objective(f, g [, sg, sgi, h])

    Creates an objective function object with:
      f:     w  ->   f(w)
      g:     w  -> ∇ f(w)
     fs:     w  ->   f(w) (stochastic with random index)
    fsi: (w, i) ->   f(w) (stochastic with index i)
     sg:     w  -> ∇ f(w) (stochastic with random index)
    sgi: (w, i) -> ∇ f(w) (stochastic with index i)
      h:     w  -> ∇²f(w)

    If sf is not provided, f is used for sf (sfi respectively).
    If sg is not provided, g is used for sg (sgi respectively).
    If h is not provided, an identity matrix is used.
"""
type Objective
      f::Function
      g::Function
     sf::Function
    sfi::Function
     sg::Function
    sgi::Function
      h::Function
    function Objective(f::Function, g::Function;
                       sf::Function = f,
                       sfi::Function = (w, i) -> sf(w),
                       sg::Function = g,
                       sgi::Function = (w, i) -> sg(w),
                       h::Function = (w -> eye(length(w))))
        new(f, g, sf, sfi, sg, sgi, h)
    end
end


"""
    Random index generator for stochastic gradient.
"""
sgr(y) = rand(1:length(y))



# include ./obj/*
LsoBase.includedir(dirname(@__FILE__)*"/obj")



end