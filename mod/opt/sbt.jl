import Obj.Objective

"""
    sbt(obj, w, s, i [, fw, gw; c, α0, η, maxiter])

    Performs stochastic Backtracking Line Search on given objective function
    for given point w and step s.
    The stepsize satisfying Armijo is returned, as well as the iteration counter.
    If however maxiter is reached, that stepsize is returned with the iteration
    counter.
"""
@fastmath function sbt(obj::Objective, w::Array{Float64,1}, s::Array{Float64,1}, i,
            fw::Float64=obj.sf(w, i), gw::Array{Float64,1}=obj.sg(w, i);
            c::Float64=1e-1, α_0::Float64=1.0, η::Float64=0.1, maxiter::Int32=10)
    @inbounds gws = (gw'*s)[1]
    for k = 1:maxiter
        if obj.sf(w + α_0*s, i) <= fw + c*α_0*gws # Armijo satisfied?
            return α_0, k-1
        else
            α_0 = η*α_0
        end
    end
    return α_0, maxiter
end