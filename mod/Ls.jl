module Ls


import LsoBase
import Obj
import Obj.Objective


"""
    LineSearch(name, strategy)

    Creates an LineSearch object for the given strategy method.
    The strategy method should have the following signature:

    (w, s, obj, b, fw, gw) -> (α, iter)

    Where:
       w: The current point 
       s: The current step direction
     obj: The objective function
       b: Index batch to evaluate objective function on. May be ignored by
          the implementation.
      fw: f(w) (present for performance reasons)
      gw: g(w) (present for performance reasons)
       α: The resulting step size
    iter: The number of LS iterations required
"""
type LineSearch
        name::ASCIIString
    strategy::Function
end


"""
    ls(linesearch, w, s [, obj, b, fw, gw])

    Performs a Line Search with the given strategy for current point w and step
    direction s. When the LS strategy depends on an objective function value, that
    function should be specified using the obj parameter.

    Optionally, b can specify a batch of indices to evaluate the objective function
    on. For performance reasons, f(w) and g(w) can be provided as fw and gw.

    This helper function should ease the access to objects of type Ls.LineSearch.
"""
function ls(linesearch::LineSearch, w::Array{Float64,1}, s::Array{Float64,1},
            obj::Union{Objective,Void}=nothing, b::Array{Int,1}=Int[],
            fw::Float64=NaN, gw::Array{Float64,1}=Float64[])

    # sanitize arguments
    if obj != nothing
      if isnan(fw)
          if length(b) == 0
              fw = Obj.f(obj, w)
          else
              fw = Obj.f(obj, w, b)
          end
      end
      if length(gw) == 0
          if length(b) == 0
              gw = Obj.g(obj, w)
          else
              gw = Obj.g(obj, w, b)
          end
      end
    end

    # run ls
    return linesearch.strategy(w, s, obj, b, fw, gw)
    
end


# include ./opt/*
LsoBase.includedir(dirname(@__FILE__)*"/ls")


end
