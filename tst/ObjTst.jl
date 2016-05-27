# 
# Tests for module Obj
# 
module ObjTst

using Base.Test
import Obj


"""
    tst()

    Run all tests on module Obj.
"""
function tst()
    println("Running all tests on module Obj...")
    @test tst_lr_sg()
    println("")
end


"""
    tst_lr_sg()

    Test Obj.lr_sg()
"""
function tst_lr_sg()
    # init
    n = 10
    m = 100
    X = randn(m,n)
    y = randn(m)
    w = randn(n)

    # tst
    g = Obj.lr_g(w, X, y)
    sg = (@parallel (+) for i = 1:m
        Obj.lr_sg(w, X, y, i)
    end) / m

    # assert
    success = (g ≈ sg) # approx equal?
    if (success)
        println("tst_lr_sg SUCCEEDED: Complete grad at once matches complete grad using lr_sg")
    else
        println("tst_lr_sg FAILED!\nComplete grad at once = $g\nComplete grad using lr_sg = $sg")
    end
    return success
end


end