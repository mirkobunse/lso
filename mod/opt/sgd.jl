import Obj
import Obj.Objective
import Ls
import Ls.LineSearch


"""
    sgd(obj, w [, ls; batchsize, ϵ, maxiter, storeiter, maxtime])

    Performs Stochastic (Sub)Gradient Descent on objective function with
    initial w and the given Line Search function. Returns info DataFrame.
"""
@fastmath function sgd(obj::Objective, w::Array{Float64,1}, ls::LineSearch=Ls.sbt(obj); batchsize::Int32=1,
             ϵ::Float64=1e-6, maxiter::Int32=1000, storeiter::Int32=100, maxtime::Int32=60)

    inf = LsoBase.new_inf()

    # print info header
    headline = @sprintf "\n%6s | %6s | %3s | %9s | %9s"  "k" "sec" "i" "f" "infeas"
    println(headline, "\n", repeat("-", length(headline)))

    # optimization
    lsiter::Int32 = 0
    start = Base.time()
    time = 0.0
    minopt = Inf
    maxf   = -Inf
    try 

        for k = 1:maxiter

            # stochastic batch setup
            b = Obj.randbatch(obj, batchsize)   # random sgd index batch
            fw = Obj.f(obj, w, b)
            gw = Obj.g(obj, w, b)

            # obtain opt, push info to array
            opt = vecnorm(gw, Inf)

            # update time
            time = Base.time() - start
            if time > maxtime
                break
            end

            # store and print info
            maxf =   max(fw,  maxf)
            minopt = min(opt, minopt)
            if (k-1)%storeiter == 0
                println(@sprintf "%6d | %6.3f | %3d | %9.3e | %9.3e"  k-1 time lsiter maxf minopt)
                LsoBase.push_inf!(inf, w, maxf, minopt, k-1, lsiter, time)
                minopt = Inf
                maxf   = -Inf
            end 

            # take step or stop
            if opt < ϵ # stopping criterion satisfied?
                break
            else
                s = -gw # -g(w)
                α, lsiter = Ls.ls(ls, w, s, b, fw, gw)
                w += α*s
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