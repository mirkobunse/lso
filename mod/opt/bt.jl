import Obj
import Obj.Objective

"""
    bt(obj, w, s [, fw, gw; c, α0, η, maxiter])

    Performs Backtracking (Armijo) Line Search on given objective function
    for given point w and step s.
    The stepsize satisfying Armijo is returned, as well as the iteration counter.
    If however maxiter is reached, that stepsize is returned with the iteration
    counter.
"""
@fastmath function bt(obj::Objective, w::Array{Float64,1}, s::Array{Float64,1},
            fw::Float64=Obj.f(obj, w), gw::Array{Float64,1}=Obj.g(obj, w);
            c::Float64=1e-1, α_0::Float64=1.0, η::Float64=0.1, maxiter::Int32=10)
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