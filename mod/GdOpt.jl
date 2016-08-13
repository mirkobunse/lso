module GdOpt


import LsoBase
import Obj
import Obj.Objective
import Ls
import Ls.LineSearch


"""
    Arbitrary state of optimizer
"""
abstract State


"""
    State for stateless optimizers
"""
type EmptyState <: State
end


"""
    GdOptimizer(init, update)

      init: Initial state.
    update: (obj, k, w, b, inf, state) -> (f(w), ∇ f(w), s, state)
"""
type GdOptimizer
      init::State
    update::Function
end


"""
    gd(obj, w [, ls; ϵ, maxiter, storeiter, maxtime])

    Performs Steepest Descent on objective function with initial w and the
    given Line Search function. Returns info DataFrame.
"""
@fastmath function opt(optimizer::GdOptimizer, ls::LineSearch, obj::Objective, w_0::Array{Float64,1};
                         ϵ::Float64=1e-6, maxiter::Int32=typemax(Int32), maxtime::Float64=60.0, batchsize::Int32=-1)
    inf = LsoBase.new_inf()

    # print info header
    headline = @sprintf "\n%6s | %6s | %3s | %9s | %9s"  "k" "sec" "i" "f" "infeas"
    println(headline, "\n", repeat("-", length(headline)))

    # optimization
    lsiter      = 0
    start       = Base.time()
    storagetime = 0.0
    minopt      = Inf
    maxf        = -Inf
    state       = optimizer.init
    try 

        for k = 1:maxiter

            b = Int32[]
            if batchsize > 0
                b = Obj.randbatch(obj, batchsize)   # random sbt index batch
            end
            fw, gw, s, state = optimizer.update(obj, k, w_0, b, inf, state)

            # obtain opt, push info to array
            opt = vecnorm(gw, Inf)

            # store and print info
            time   = Base.time() - start
            maxf   = max(fw,  maxf)
            minopt = min(opt, minopt)
            if time - storagetime > .25 || k == 1 || time > maxtime
                println(@sprintf "%6d | %6.3f | %3d | %9.3e | %9.3e"  k-1 time lsiter maxf minopt)
                LsoBase.push_inf!(inf, w_0, maxf, minopt, k-1, lsiter, time)
                storagetime = floor(time*4) / 4
                minopt = Inf
                maxf   = -Inf
            end

            # take step or stop
            if time > maxtime # max time over?
                break
            elseif opt < ϵ # stopping criterion satisfied?
                break
            else
                α, lsiter = Ls.ls(ls, w_0, s, b, fw, gw)
                w_0 += α*s
            end

        end # end of optimization

    catch e

        if isa(e, InterruptException)
            println("Optimization aborted due to InterruptException")
        else
            throw(e)
        end

    end

    return inf
end


# include ./opt/*
LsoBase.includedir(dirname(@__FILE__)*"/gdopt")


end