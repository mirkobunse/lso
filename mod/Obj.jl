module Obj


import LsoBase


"""
    Objective(f, g [, h; f_b, g_b])

    Creates an objective function object with:
      f:     w  ->   f(w)
      g:     w  -> ∇ f(w)
      h:     w  -> ∇²f(w)
    f_b: (w, b) ->   f(w) (stochastic with indices b)
    g_b: (w, b) -> ∇ f(w) (stochastic with indices b)
    dim: Number of evaluations needed in order to compute
         full function value / gradient, e.g., the number
         of examples

    If f_b is not provided, f is used for f_b.
    If g_b is not provided, g is used for g_b.
    If h is not provided, an identity matrix is used.
"""
type Objective
      f::Function
      g::Function
      h::Function
    f_b::Function
    g_b::Function
    dim::Int
    function Objective(f::Function, g::Function,
                       h::Function = (w -> eye(length(w)));
                       f_b::Function = (w, b) -> f(w),
                       g_b::Function = (w, b) -> g(w),
                       dim::Int = -1)
        new(f, g, h, f_b, g_b, dim)
    end
end


"""
    f(obj, w [, b])

    Returns the function value of the objective function obj at point w.
    Optionally, use the index batch b to evaluate the function against.

    This helper function should ease the access to objects of type Obj.Objective.
"""
function f(obj::Objective, w::Array{Float64,1}, b::Array{Int,1}=Int[])
  if length(b) == 0
    return obj.f(w)
  else
    return obj.f_b(w, b)
  end
end


"""
    g(obj, w [, b])

    Returns the gradient of the objective function obj at point w.
    Optionally, use the index batch b to evaluate the function against.

    This helper function should ease the access to objects of type Obj.Objective.
"""
function g(obj::Objective, w::Array{Float64,1}, b::Array{Int,1}=Int[])
  if length(b) == 0
    return obj.g(w)
  else
    return obj.g_b(w, b)
  end
end


"""
    randbatch(obj, batchsize)

    Returns a random index batch of the given size for the objective function.
"""
function randbatch(obj::Objective, batchsize::Int=1)
  if batchsize == 1
    if obj.dim > 1
      return [rand(1:obj.dim)]
    else
      return Int[]
    end
  else
    if obj.dim < batchsize
      batchsize = obj.dim
    end
    return randperm(obj.dim)[1:batchsize]
  end
end


# include ./obj/*
LsoBase.includedir(dirname(@__FILE__)*"/obj")



end
