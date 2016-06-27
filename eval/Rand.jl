# 
module Rand

import LsoBase
import GD
import LinReg


"""
    gd_bt_rand([; maxiter, n, m])

    Test GD.gd_bt() with random data
"""
function gd_bt(maxiter=1000; n=1000, m=10000)
    # init random data
    w_true = randn(n)
    X = randn(m,n)
    y = X*w_true
    f(w) = LinReg.f(w, X, y)
    g(w) = LinReg.g(w, X, y)

    # tst
    w0 = randn(n)
    @time inf = GD.gd_bt(f, g, w0, maxiter=maxiter)
    w = inf[:w][end]
    iter = inf[:iter][end]
    opt = inf[:opt][end]

    # assert
    success = isapprox(w, w_true, atol=1e-4)
    if (success)
        println("gd_bt_rand SUCCEEDED: Found true w in $iter steps (Optimality $opt)")
    else
        println("gd_bt_rand FAILED: Did not find true w in $iter steps (Optimality $opt).\nw = $w\nw_true = $w_true")
    end

    return inf
end


end