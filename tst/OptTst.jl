# 
# Tests for module Opt
# 
module OptTst

using Base.Test
import Opt
import Obj


"""
    tst()

    Run all tests on module Opt.
"""
function tst()
    println("Running all tests on module Opt...")
    @test tst_ls_bt()
    println("")
end


"""
    tst_ls_bt()

    Test Opt.ls_bt()
"""
function tst_ls_bt()
    maxiter = 100

    # init intractable
    f(w) = w[1]^2
    g(w) = [2w[1], 0]
    w = [0, 0]
    s = [1, 0]
    # assert
    α, iter1 = Opt.ls_bt(f, g, w, s, maxiter=maxiter)
    success1 = (iter1 == maxiter)

    # init tractable
    f(w) = -w[1]^2
    g(w) = [-2w[1], 0]
    w = [0, 0]
    s = [1, 0]
    # assert
    α, iter2 = Opt.ls_bt(f, g, w, s, maxiter=maxiter)
    success2 = (iter2 < maxiter)

    # print results
    if (success1 && success2)
        println("tst_ls_bt SUCCEEDED: Intractable LS required max number of iterations, tractable LS required less.")
    elseif (!success1)
        println("tst_ls_bt FAILED: Intractable LS required less than max number of iterations ($iter1)!")
    elseif (!success2)
        println("tst_ls_bt FAILED: Tractable LS required max number of iterations!")
    end

    return success1 && success2
end


"""
    tst_gd_bt_rosenbrock([maxiter])

    Test Opt.gd_bt() with the Rosenbrock function
"""
function tst_gd_bt_rosenbrock(maxiter=10000)
    # init rosenbrock function
    f(w) = 100*(w[2]-w[1]^2)^2 + (1-w[1])^2
    function g(w)
        g2 = 200*(w[2]-w[1]^2)
        return [-2*(w[1]*g2 + (1-w[1])), g2]
    end
    # tst
    w = zeros(2)
    Opt.gd_bt(f, g, w, maxiter=maxiter)
end



"""
    tst_gd_bt_rand([; maxiter, n, m])

    Test Opt.gd_bt() with random data
"""
function tst_gd_bt_rand(maxiter=1000; n=1000, m=10000)
    # init random data
    w_true = randn(n)
    X = randn(m,n)
    y = X*w_true
    f(w) = Obj.lr_f(w, X, y)
    g(w) = Obj.lr_g(w, X, y)

    # tst
    w0 = randn(n)
    w, opt, iter = Opt.gd_bt(f, g, w0, maxiter=maxiter)

    # assert
    if (isapprox(w, w_true, atol=.333))
        println("tst_gd_bt SUCCEEDED: Found true w in $iter steps (Optimality $opt)")
    else
        println("tst_gd_bt FAILED: Did not find true w in $iter steps (Optimality $opt).\n     w = $w\nw_true = $w_true")
    end
end


end