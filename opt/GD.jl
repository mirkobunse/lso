#
# Module for optimization algorithms
#
module GD


# export ...

import LsoBase


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


"""
    gd_bt(f, g, w [, ls; ϵ, maxiter, printiter])

    Performs Steepest Descent with the given Line Search. Returns array
    of IterInfo.

    Function f and its gradient g should only depend on the argument w.
    The parameter w is used as initial point.
"""
function gd(f::Function, g::Function, w::Array{Float64,1}, ls::Function=bt;
            ϵ::Float64=1e-6, maxiter::Int32=1000, printiter::Int32=5)
    inf = LsoBase.new_inf()

    # print info header
    headline = @sprintf "\n%6s | %3s | %9s | %9s"  "k" "i" "f" "opt"
    println(headline, "\n", repeat("-", length(headline)))

    # optimization
    lsiter = 0
    for k = 1:maxiter

        fw = f(w)
        gw = g(w)

        # obtain opt, push info to array
        opt = vecnorm(gw, Inf)
        LsoBase.push_inf!(inf, w, fw, opt, k-1, lsiter)

        # print info
        if (k-1)%printiter == 0
            println(@sprintf "%6d | %3d | %9.3e | %9.3e"  k-1 lsiter fw opt)
        end 

        # take step or stop
        if opt < ϵ # stopping criterion satisfied?
            break
        else
            s = -gw # -g(w)
            α, lsiter = ls(f, g, w, s, fw, gw)
            w += α*s
        end

    end # end of optimization

    return inf
end


end