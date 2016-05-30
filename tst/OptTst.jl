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


end