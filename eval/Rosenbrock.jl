# 
# Evaluation with Mnist data
# 
module Rosenbrock

import LsoBase
import GD


"""
    gd_bt_rosenbrock([maxiter])

    Test Opt.gd_bt() with the Rosenbrock function
"""
function gd_bt_rosenbrock(maxiter=1000000)
    # init rosenbrock function
    f(w) = 100*(w[2]-w[1]^2)^2 + (1-w[1])^2
    function g(w)
        g2 = 200*(w[2]-w[1]^2)
        return [-2*(w[1]*g2 + (1-w[1])), g2]
    end
    # tst
    w = zeros(2)
    @time GD.gd_bt(f, g, w, maxiter=maxiter, printiter=10000)
end


"""
    gd_bt_rand([; maxiter, n, m])

    Test GD.gd_bt() with random data
"""
function gd_bt_rand(maxiter=1000; n=1000, m=10000)
    # init random data
    w_true = randn(n)
    X = randn(m,n)
    y = X*w_true
    f(w) = Obj.lr_f(w, X, y)
    g(w) = Obj.lr_g(w, X, y)

    # tst
    w0 = randn(n)
    @time infarr = GD.gd_bt(f, g, w0, maxiter=maxiter)
    w = infarr[end].w
    iter = infarr[end].iter
    opt = infarr[end].opt

    # assert
    success = isapprox(w, w_true, atol=1e-4)
    if (success)
        println("tst_gd_bt SUCCEEDED: Found true w in $iter steps (Optimality $opt)")
    else
        println("tst_gd_bt FAILED: Did not find true w in $iter steps (Optimality $opt).\nw = $w\nw_true = $w_true")
    end

    return infarr
end


end