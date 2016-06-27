#
# Module for optimization algorithms
#
module GD


# export ...

import LsoBase


"""
    bt(f, g, w, s [, c, α0, η; maxiter])

    Performs Backtracking (Armijo) Line Search for given point w and step s.

    Function f and its gradient g should only depend on the argument w.
    The parameter α_0 is used as initial stepsize.
    The stepsize satisfying Armijo (or when maxiter is reached) is returned,
    as well as the iteration counter.
"""
function bt(f, g, w, s, c=1e-3, α_0=1, η=0.5; maxiter=20, fw=f(w), gw=g(w))
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
    gd_bt(f, g, w [, ϵ, c, α_0, η; maxiter, maxbtiter])

    Performs Steepest Descent with Backtracking Line Search. Returns array
    of IterInfo.

    Function f and its gradient g should only depend on the argument w.
    The parameter w is used as initial point.
"""
function gd_bt(f, g, w, ϵ=1e-6, c=1e-3, α_0=1, η=0.5; maxiter=1000, maxbtiter=20, printiter=1)
    inf = LsoBase.newinf()

    # print info header
    headline = @sprintf "%6s | %3s | %9s | %9s"  "k" "i" "f" "opt"
    println(headline, "\n", repeat("-", length(headline)))

    # optimization
    lsiter = 0
    for k = 1:maxiter

        fw = f(w)
        gw = g(w)

        # obtain opt, push info to array
        opt = vecnorm(gw, Inf)
        LsoBase.pushinf!(inf, w, opt, k-1, lsiter)

        # print info
        if (k-1)%printiter == 0
            println(@sprintf "%6d | %3d | %9.3e | %9.3e"  k-1 lsiter fw opt)
        end 

        # take step or stop
        if opt < ϵ # stopping criterion satisfied?
            break
        else
            s = -gw # -g(w)
            α, lsiter = bt(f, g, w, s, c, α_0, η, maxiter=maxbtiter, fw=fw, gw=gw)
            w += α*s
        end

    end # end of optimization

    return inf
end


end