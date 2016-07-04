import Obj
import Opt
import Plotting

"""
    rand_gd_bt([maxiter; n, m])

    Test GD with BT on Linear Regression of random data.
"""
function rand_gd_bt(maxiter=1000; n=100, m=1000)
    # init random data
    w_true = randn(n)
    X = randn(m,n)
    y = X*w_true

    # tst
    w0 = randn(n)
    @time inf = Opt.gd(Obj.logreg(X, y), w0, maxiter=maxiter)
    w = inf[end, :w]
    iter = inf[end, :iter]
    opt = inf[end, :opt]

    # assert
    success = isapprox(w, w_true, atol=1e-3)
    if (success)
        println("\nSUCCEEDED: Found true w in $iter steps!")
    else
        println("\nFAILED: Did not find true w in $iter steps!")
    end
    println(@sprintf "%16s = %9.3e\n%16s = %9.3e" "‖∇f(x)‖∞" opt "‖w-w_true‖ " vecnorm(w-w_true, 2))

    # ask user for plot
    print("\nPlot progress? (y/N): ")
    if startswith(readline(STDIN), "y")
        Plotting.display_inf(inf)
    end
    return nothing # inf

end