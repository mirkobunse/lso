import Obj
import Opt
import Plotting


"""
    rand_gd_bt([maxiter; n, m])

    Test GD with BT on Linear Regression of random data.
"""
function rand_gd_bt_linreg(maxiter=1000; n=100, m=1000)
    # init random data
    w_true = randn(n)
    X = randn(m,n)
    y = X*w_true

    # tst
    w0 = randn(n)
    @time inf = Opt.gd(Obj.linreg(X, y), w0, maxiter=maxiter)
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


"""
    rand_gd_bt([maxiter; n, m])

    Test GD with BT on Logistic Regression of 2-class random data.
"""
function rand_gd_bt_logreg(maxiter=2500; n=100, m=1000, noisefac=4.8)

    m2 = convert(Int32, m/2)    # m/2 as integer

    # init classes
    centers_1 = randn(n)'
    centers_2 = randn(n)'
    noise = noisefac * randn(n)'
    X_1 = randn(m2, n) .+ centers_1 .- noise
    X_2 = randn(m2, n) .+ centers_2 .- noise

    # random permutation of examples
    perm = randperm(m)
    X = vcat(X_1, X_2)[perm, :]
    y = vcat(vec(repeat([-1.0], inner=[m2,1])), vec(repeat([1.0], inner=[m2,1])))[perm]

    # tst
    w0 = randn(n)
    @time inf = Opt.gd(Obj.linreg(X, y), w0, maxiter=maxiter, printiter=100)
    w = inf[end, :w]
    iter = inf[end, :iter]
    opt = inf[end, :opt]

    # acc
    acc = LsoBase.acc(y, Obj.logreg_predict(w, X))
    println("\nTraining set accuracy: $acc")

    # ask user for plot
    print("\nPlot progress? (y/N): ")
    if startswith(readline(STDIN), "y")
        Plotting.display_inf(inf)
    end
    return nothing # inf

end