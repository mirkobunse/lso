import Obj

function linreg_sg()
    println("Testing Obj.linreg(X, y).sg(w)...")

    # init
    n = 10
    m = 100
    X = randn(m,n)
    y = randn(m)
    w = randn(n)

    # tst
    g = Obj.linreg(X, y).g(w)
    sg = sum([ Obj.linreg(X, y).sgi(w, i) for i = 1:m ]) ./ m

    # assert
    success = (g ≈ sg) # approx equal?
    if (success)
        println("SUCCEEDED: Complete grad at once matches complete grad using sg_linreg")
    else
        println("FAILED! 2norm-distance between grad and stochastic grad is $(vecnorm(g-sg, 2))")
    end
    return nothing
end

function logreg_sg()
    println("Testing Obj.logreg(X, y).sg(w)...")

    # init
    n = 10
    m = 100
    X = randn(m,n)
    y = randn(m)
    w = randn(n)

    # tst
    g = Obj.logreg(X, y).g(w)
    sg = sum([ Obj.logreg(X, y).sgi(w, i) for i = 1:m ]) ./ m

    # assert
    success = (g ≈ sg) # approx equal?
    if (success)
        println("SUCCEEDED: Complete grad at once matches complete grad using sg_logreg")
    else
        println("FAILED! 2norm-distance between grad and stochastic grad is $(vecnorm(g-sg, 2))")
    end
    return nothing
end