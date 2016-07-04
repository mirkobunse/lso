import Opt
import Obj
import Plotting

"""
    rosenbrock_gd_bt([maxiter])

    Test GD with BT on the Rosenbrock function.
"""
function rosenbrock_gd_bt(maxiter::Int32=100000) # 1e5

    # init rosenbrock function
    rosenbrock = Obj.Objective(
        w -> 100*(w[2]-w[1]^2)^2 + (1-w[1])^2,
        function (w)
            g2 = 200*(w[2]-w[1]^2)
            return [-2*(w[1]*g2 + (1-w[1])), g2]
        end
    )

    # tst
    w = zeros(2)
    inf = @time Opt.gd(rosenbrock, w, maxiter=maxiter, printiter=10000)

    return nothing # inf
    
end