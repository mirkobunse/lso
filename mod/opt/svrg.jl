import Obj.Objective

"""
    svrg(obj, w [, ls; batchsize, estimation, strategy, ϵ, maxiter, printiter])

    Performs SVRG (Stochastic Variance-Reduced Gradient Descent) on objective function with
    initial w and the given Line Search function. Returns info DataFrame.

    The estimation parameter describes the number of iterations before a new w estimate is
    obtained. The strategy to obtain a new w estimate is set by the strategy parameter. The
    following stategies exist:
    - :last     Snapshat of very last iteration (default)
    - :rand     Random snapshot of last iterations
    - :avg      Average of last iterations
"""
@fastmath function svrg(obj::Objective, w::Array{Float64,1}, ls::Function=sbt;
             batchsize::Int32=1, estimation::Int32=10, strategy::Symbol=:last,
             ϵ::Float64=1e-6, maxiter::Int32=1000, timeiter::Int32=100)

    inf = LsoBase.new_inf()

    # print info header
    headline = @sprintf "\n%6s | %6s | %3s | %9s | %9s"  "k" "sec" "i" "f" "infeas"
    println(headline, "\n", repeat("-", length(headline)))

    # optimization
    lsiter::Int32 = 0
    start = Base.time()
    time = 0.0
    w_est = nothing  # w estimate (will be set on first iteration)
    g_est = nothing  # μ (= full gradient of w estimate)
    try 

        for k = 1:maxiter

            # stochastic batch setup
            i = obj.rng(batchsize)   # random sgd index batch
            fw = obj.sf(w, i)
            gw = obj.sg(w, i)

            # svrg estimation update
            if (k-1) % estimation == 0
                if strategy == :last
                    w_est  = w       # TODO implement other w choices
                elseif strategy == :rand
                    w_est = inf[:w][ end-rand(1:estimation)+1 ]
                elseif strategy == :avg
                    w_est = mean(inf[:w][ (end-estimation+1):end ], 2)
                end
                g_est = obj.g(w_est)
            end
            sg_est = obj.sg(w_est, i)   # stochastic gradient of w estimate

            # obtain opt, push info to array
            opt = vecnorm(gw, Inf)

            # update time and print info
            if (k-1) % timeiter == 0
                if k > 0
                    time = Base.time() - start
                else
                    start = Base.time()
                    time = 0.0
                end
                println(@sprintf "%6d | %6.3f | %3d | %9.3e | %9.3e"  k-1 time lsiter fw opt)
            end 

            LsoBase.push_inf!(inf, w, fw, opt, k-1, lsiter, time)

            # take step or stop
            if opt < ϵ # stopping criterion satisfied?
                break
            else
                s = -gw + sg_est - g_est # -g(w) + g(w_est) - μ
                α, lsiter = ls(obj, w, s, i, fw, gw)
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