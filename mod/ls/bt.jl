import Obj
import Obj.Objective


"""
    bt([; c, α_0, η, maxiter])

    Returns a LineSearch strategy for Backtracking (Armijo) Line Search
    with given parametrization:

          c: Armijo condition constant (default 1e-1)
        α_0: Initial step size value (default 1.0)
          η: Decrease rate (default: 0.1)
    maxiter: Max number of LS iterations (default: 10)
"""
function bt(; c::Float64=1e-1, α_0::Float64=1.0, η::Float64=0.1, maxiter::Int32=10)

    return LineSearch(
        # name
        "bt",

        # strategy
        function (w::Array{Float64,1}, s::Array{Float64,1}, obj::Union{Objective,Void},
                  b::Array{Int32,1}, fw::Float64, gw::Array{Float64,1})
            return _bt(w, s, obj, fw, gw, c, α_0, η, maxiter)   # b is ignored
        end
    )

end


@fastmath function _bt(w::Array{Float64,1}, s::Array{Float64,1}, obj::Objective,
                       fw::Float64, gw::Array{Float64,1},
                       c::Float64, α_0::Float64, η::Float64, maxiter::Int32)

    @inbounds gws = (gw'*s)[1]
    for i = 1:maxiter

        if Obj.f(obj, w + α_0*s) <= fw + c*α_0*gws # Armijo satisfied?
            return α_0, i-1
        else
            α_0 = η*α_0
        end

    end
    return α_0, maxiter

end