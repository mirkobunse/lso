import Obj
import Obj.Objective
import Ls
import Ls.LineSearch


"""
    svrg(obj, w [, ls; batchsize, estimation, strategy, ϵ, maxiter, storeiter, maxtime])

    Performs SVRG (Stochastic Variance-Reduced Gradient Descent) on objective function with
    initial w and the given Line Search function. Returns info DataFrame.

    The estimation parameter describes the number of iterations before a new w estimate is
    obtained. The strategy to obtain a new w estimate is set by the strategy parameter. The
    following stategies exist:
    - :last     Snapshat of very last iteration (default)
    - :rand     Random snapshot of last iterations
    - :avg      Average of last iterations
"""
@fastmath function svrg(obj::Objective, w::Array{Float64,1}, ls::LineSearch=Ls.sbt(obj);
             batchsize::Int32=1, estimation::Int32=10, strategy::Symbol=:last,
             ϵ::Float64=1e-6, maxiter::Int32=typemax(Int32), maxtime::Int32=60)

    inf = LsoBase.new_inf()

    # print info header
    headline = @sprintf "\n%6s | %6s | %3s | %9s | %9s"  "k" "sec" "i" "f" "infeas"
    println(headline, "\n", repeat("-", length(headline)))

    # optimization
    lsiter::Int32 = 0
    start = Base.time()
    storagetime = 0.0
    minopt = Inf
    maxf   = -Inf
    w_est = nothing  # w estimate (will be set on first iteration)
    gw_est = nothing  # μ (= full gradient of w estimate)
    try 

        for k = 1:maxiter

            # stochastic batch setup
            b = Obj.randbatch(obj, batchsize)   # random sgd index batch
            fw = Obj.f(obj, w, b)
            gw = Obj.g(obj, w, b)

            # obtain opt, push info to array
            opt = vecnorm(gw, Inf)

            # store and print info
            time   = Base.time() - start
            maxf   = max(fw,  maxf)
            minopt = min(opt, minopt)
            if time - storagetime > .25 || k == 1 || time > maxtime
                println(@sprintf "%6d | %6.3f | %3d | %9.3e | %9.3e"  k-1 time lsiter maxf minopt)
                LsoBase.push_inf!(inf, w, maxf, minopt, k-1, lsiter, time)
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

                # svrg estimation update
                if (k-1) % estimation == 0
                    if strategy == :last
                        w_est  = w       # TODO implement other w choices
                    elseif strategy == :rand
                        w_est = inf[:w][ end-rand(1:estimation)+1 ]
                    elseif strategy == :avg
                        w_est = mean(inf[:w][ (end-estimation+1):end ], 2)
                    end
                    gw_est = Obj.g(obj, w_est)
                end
                gw_est_b = Obj.g(obj, w_est, b)   # stochastic gradient of w estimate

                s = -gw + gw_est_b - gw_est # -g(w) + g(w_est) - μ
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