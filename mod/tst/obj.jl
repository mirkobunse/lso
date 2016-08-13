import Obj
import Obj.Objective

function linreg_g()
    println("\nTesting gradient of Obj.linreg(X, y)...")
    X, y, w = _obj_randXyw() # init
    _obj_g(Obj.linreg(X, y), w)
end

function logreg_g()
    println("\nTesting gradient of Obj.logreg(X, y)...")
    X, y, w = _obj_randXyw() # init
    _obj_g(Obj.logreg(X, y), w)
end



_obj_randXyw(n=10, m=100) = randn(m,n), randn(m), randn(n)

function _obj_g(obj::Objective, w::Array{Float64,1})
    g = Obj.g(obj, w)
    g_b = sum([ Obj.g(obj, w, [i]) for i = 1:obj.dim ]) ./ obj.dim

    # assert
    success = (g ≈ g_b) # approx equal?
    if (success)
        println("SUCCEEDED: Complete grad at once matches complete grad using g_b")
    else
        println("FAILED! 2norm-distance between grad and stochastic grad is $(vecnorm(g-g_b, 2))")
    end
    return nothing
end