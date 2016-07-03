import Opt
import Obj

function bt()
    println("Testing Opt.bt...")
    maxiter = 100

    # init intractable
    f(w) = w[1]^2
    g(w) = [2w[1], 0]
    w = [0.0, 0.0]
    s = [1.0, 0.0]
    # assert
    α, iter1 = Opt.bt(f, g, w, s, maxiter=maxiter)
    success1 = (iter1 == maxiter)

    # init tractable
    f(w) = -w[1]^2
    g(w) = [-2w[1], 0]
    w = [0.0, 0.0]
    s = [1.0, 0.0]
    # assert
    α, iter2 = Opt.bt(f, g, w, s, maxiter=maxiter)
    success2 = (iter2 < maxiter)

    # print results
    if (success1 && success2)
        println("SUCCEEDED: Intractable LS required max number of iterations, tractable LS required less.")
    elseif (!success1)
        println("FAILED: Intractable LS required less than max number of iterations ($iter1)!")
    elseif (!success2)
        println("FAILED: Tractable LS required max number of iterations!")
    end

    return nothing
end