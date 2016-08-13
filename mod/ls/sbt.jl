import Obj
import Obj.Objective


"""
    sbt(obj [; c, α_0, η, maxiter])

    Returns a LineSearch strategy for stochastic Backtracking Line Search on given
    objective function with given parametrization:

          c: Armijo condition constant (default 1e-1)
        α_0: Initial step size value (default 1.0)
          η: Decrease rate (default: 0.1)
    maxiter: Max number of LS iterations (default: 10)
"""
function sbt(obj::Objective; c::Float64=1e-1, α_0::Float64=1.0, η::Float64=0.1, maxiter::Int32=10)
    return LineSearch(
        obj, # objective function

        # strategy
        function (w::Array{Float64,1}, s::Array{Float64,1}, b::Array{Int32,1}, fw::Float64, gw::Array{Float64,1})
            return _sbt(obj, w, s, b, fw, gw, c, α_0, η, maxiter)
        end
    )
end


@fastmath function _sbt(obj::Objective, w::Array{Float64,1}, s::Array{Float64,1},
                        b::Array{Int32,1}, fw::Float64, gw::Array{Float64,1},
                        c::Float64, α_0::Float64, η::Float64, maxiter::Int32)
    @inbounds gws = (gw'*s)[1]
    for i = 1:maxiter

        # function value at step
        fα = -1.0
        if length(b) == 0
            fα = Obj.f(obj, w + α_0*s)
        else
            fα = Obj.f(obj, w + α_0*s, b)
        end

        # Armijo satisfied?
        if fα <= fw + c*α_0*gws
            return α_0, i-1
        else
            α_0 = η*α_0
        end

    end
    return α_0, maxiter
end