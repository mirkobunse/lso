import Obj
import Obj.Objective
import Ls
import Ls.LineSearch


"""
    gd(obj, w [, ls; ϵ, maxiter, storeiter, maxtime])

    Performs Steepest Descent on objective function with initial w and the
    given Line Search function. Returns info DataFrame.
"""
@fastmath function gd(obj::Objective, w::Array{Float64,1}, ls::LineSearch=Ls.bt(obj); batchsize::Int32=-1,
            ϵ::Float64=1e-6, maxiter::Int32=typemax(Int32), maxtime::Int32=60)
    inf = LsoBase.new_inf()

    # print info header
    headline = @sprintf "\n%6s | %6s | %3s | %9s | %9s"  "k" "sec" "i" "f" "infeas"
    println(headline, "\n", repeat("-", length(headline)))

    # optimization
    lsiter = 0
    start = Base.time()
    storagetime = 0.0
    minopt = Inf
    maxf   = -Inf
    try 

        for k = 1:maxiter

            fw = Obj.f(obj, w)
            gw = Obj.g(obj, w)

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
                s = -gw # -g(w)
                b = Inf32[]
                if batchsize > 0
                    b = Obj.randbatch(obj, batchsize)   # random sbt index batch
                end
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