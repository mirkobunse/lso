import Obj

function sg_linreg()
    println("Testing Obj.sg_linreg...")

    # init
    n = 10
    m = 100
    X = randn(m,n)
    y = randn(m)
    w = randn(n)

    # tst
    g = Obj.g_linreg(w, X, y)
    sg = (@parallel (+) for i = 1:m
        Obj.sg_linreg(w, X, y, i)
    end) / m

    # assert
    success = (g ≈ sg) # approx equal?
    if (success)
        println("SUCCEEDED: Complete grad at once matches complete grad using sg_linreg")
    else
        println("FAILED! 2norm-distance between grad and stochastic grad is $(vecnorm(g-sg, 2))")
    end
    return nothing
end