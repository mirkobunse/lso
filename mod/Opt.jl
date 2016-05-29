#
# Module for optimization algorithms
#
module Opt


# export ...


const printiter = 1 # when iter%printiter == 0, print iteration info


type IterInfo
    w::Array{Float64, 1}
    opt::Float64
    iter::Integer
    lsiter::Integer
end


"""
    ls_bt(f, g, w, s [, c, α0, η; maxiter])

    Performs Backtracking (Armijo) Line Search for given point w and step s.

    Function f and its gradient g should only depend on the argument w.
    The parameter α0 is used as initial stepsize.
    The stepsize satisfying Armijo (or when maxiter is reached) is returned,
    as well as the iteration counter.
"""
function ls_bt(f, g, w, s, c=1e-3, α0=1, η=0.5; maxiter=20, fw=f(w), gw=g(w))
    gws = (gw'*s)[1]
    for i = 1:maxiter
        if f(w + α0*s) <= fw + α0*gws # Armijo satisfied?
            return α0, i-1
        else 
            α0 = η*α0
        end
    end
    return α0, maxiter
end


"""
    gd_bt(f, g, w [, ϵ, c, α0, η; maxiter, maxbtiter])

    Performs Steepest Descent with Backtracking Line Search. Returns array
    of IterInfo.

    Function f and its gradient g should only depend on the argument w.
    The parameter w is used as initial point.
"""
function gd_bt(f, g, w, ϵ=1e-6, c=1e-3, α0=1, η=0.5; maxiter=1000, maxbtiter=20, printiter=Opt.printiter)
    infarr::Array{IterInfo, 1} = []

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
        push!(infarr, IterInfo(w, opt, k-1, lsiter))

        # print info
        if (k-1)%printiter == 0
            println(@sprintf "%6d | %3d | %9.3e | %9.3e"  k-1 lsiter fw opt)
        end 

        # take step or stop
        if opt < ϵ # stopping criterion satisfied?
            break
        else
            s = -gw # -g(w)
            α, lsiter = ls_bt(f, g, w, s, c, α0, η, maxiter=maxbtiter, fw=fw, gw=gw)
            w += α*s
        end

    end # end of optimization

    return infarr
end


end