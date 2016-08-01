import Obj.Objective

"""
    gd(obj, w [, ls; ϵ, maxiter, timeiter, maxtime])

    Performs Steepest Descent on objective function with initial w and the
    given Line Search function. Returns info DataFrame.
"""
@fastmath function gd(obj::Objective, w::Array{Float64,1}, ls::Function=bt;
            ϵ::Float64=1e-6, maxiter::Int32=1000, timeiter::Int32=5, maxtime::Int32=60)
    inf = LsoBase.new_inf()

    # print info header
    headline = @sprintf "\n%6s | %6s | %3s | %9s | %9s"  "k" "sec" "i" "f" "infeas"
    println(headline, "\n", repeat("-", length(headline)))

    # optimization
    lsiter = 0
    start = Base.time()
    time = 0.0
    try 

        for k = 1:maxiter

            fw = obj.f(w)
            gw = obj.g(w)

            # obtain opt, push info to array
            opt = vecnorm(gw, Inf)

            # update time and print info
            if (k-1)%timeiter == 0
                time = Base.time() - start
                println(@sprintf "%6d | %6.3f | %3d | %9.3e | %9.3e"  k-1 time lsiter fw opt)
                if time > maxtime
                    break
                end
            end 

            LsoBase.push_inf!(inf, w, fw, opt, k-1, lsiter, time)

            # take step or stop
            if opt < ϵ # stopping criterion satisfied?
                break
            else
                s = -gw # -g(w)
                α, lsiter = ls(obj, w, s, fw, gw)
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