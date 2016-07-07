import Obj.Objective

"""
    sgd(obj, w [, ls; batchSize, ϵ, maxiter, printiter])

    Performs Stochastic (Sub)Gradient Descent on objective function with
    initial w and the given Line Search function. Returns info DataFrame.
"""
function sgd(obj::Objective, w::Array{Float64,1}, ls::Function=bt;
             batchSize::Int32=1,
             ϵ::Float64=1e-6, maxiter::Int32=1000, printiter::Int32=100)
    inf = LsoBase.new_inf()

    # print info header
    headline = @sprintf "\n%6s | %3s | %9s | %9s"  "k" "i" "f" "infeas"
    println(headline, "\n", repeat("-", length(headline)))

    # optimization
    lsiter::Int32 = 0
    try 

        for k = 1:maxiter

            fw = obj.f(w)
            gw = sum([ obj.sg(w) for s=1:batchSize ]) ./ batchSize

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