#
# Module for optimization algorithms
#
module Opt


# export ...


"""
    ls_bt(f, g, w, s [, c, α0, η; maxiter])

    Performs Backtracking (Armijo) Line Search for given point w and step s.

    Function f and its gradient g should only depend on the argument w.
    The parameter α0 is used as initial stepsize.
    The stepsize satisfying Armijo (or when maxiter is reached) is returned,
    as well as the iteration counter.
"""
function ls_bt(f, g, w, s, c=1e-3, α0=1, η=0.5; maxiter=20)
    for i = 1:maxiter
        if f(w + α0*s) <= f(w) + (α0*g(w)'*s)[1] # Armijo satisfied?
            return α0, i-1
        else 
            α0 = η*α0
        end
    end
    return α0, maxiter
end


"""
    gd_bt(f, g, w [, ϵ, c, α0, η; maxiter, maxbtiter])

    Performs Steepest Descent with Backtracking Line Search.

    Function f and its gradient g should only depend on the argument w.
    The parameter w is used as initial point.
"""
function gd_bt(f, g, w, ϵ=1e-3, c=1e-3, α0=1, η=0.5; maxiter=1000, maxbtiter=20)
    headline = @sprintf "%6s | %3s | %9s | %9s"  "k" "i" "f" "opt"
    println(headline, "\n", repeat("-", length(headline)))
    lsiter = 0
    opt = Inf
    for k = 1:maxiter
        opt = vecnorm(g(w), Inf)
        if ((k-1)%100 == 0 || k == maxiter)
            println(@sprintf "%6d | %3d | %9.3e | %9.3e"  k-1 lsiter f(w) opt)
        end
        if opt < ϵ # stopping criterion satisfied?
            return w, k-1
        else
            s = -g(w)
            α, lsiter = ls_bt(f, g, w, s, c, α0, η, maxiter=maxbtiter)
            w += α*s
        end
    end
    return w, opt, maxiter
end


end