"""
    bt(f, g, w, s [, fw, gw; c, α0, η, maxiter])

    Performs Backtracking (Armijo) Line Search for given point w and step s.

    Function f and its gradient g should only depend on the argument w.
    The parameter α_0 is used as initial stepsize.
    The stepsize satisfying Armijo (or when maxiter is reached) is returned,
    as well as the iteration counter.
"""
function bt(f::Function, g::Function, w::Array{Float64,1}, s::Array{Float64,1},
            fw::Float64=f(w), gw::Array{Float64,1}=g(w);
            c::Float64=1e-3, α_0::Float64=1.0, η::Float64=0.5, maxiter::Int32=20)
    gws = (gw'*s)[1]
    for i = 1:maxiter
        if f(w + α_0*s) <= fw + α_0*gws # Armijo satisfied?
            return α_0, i-1
        else 
            α_0 = η*α_0
        end
    end
    return α_0, maxiter
end