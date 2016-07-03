"""
    gd(f, g, w [, ls; ϵ, maxiter, printiter])

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