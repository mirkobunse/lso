import Opt
import Obj
import Plotting

"""
    rosenbrock_gd_bt([maxiter])

    Test GD with BT on the Rosenbrock function.
"""
function rosenbrock_gd_bt(maxiter::Int32=10000) # 1e5

    # init rosenbrock function
    rosenbrock = Obj.Objective(
        w::Array{Float64,1} -> 100*(w[2]-w[1]^2)^2 + (1-w[1])^2,
        function (w::Array{Float64,1})
            g2 = 200*(w[2]-w[1]^2)
            return [-2*(w[1]*g2 + (1-w[1])), g2]
        end
    )

    # tst
    w = randn(2)
    inf = @time Opt.gd(rosenbrock, w, maxiter=maxiter, printiter=100)

    # ask user for plot
    print("\nPlot progress? (y/N): ")
    if startswith(readline(STDIN), "y")
        Plotting.display_plot(Plotting.plot_inf(inf))
    end
    return nothing # inf
    
end