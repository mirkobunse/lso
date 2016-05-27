import Obj

function tst_sg_lr()
    # init
    n = 10
    m = 100
    X = randn(m,n)
    y = randn(m)
    w = randn(n)

    # tst
    g = Obj.g_lr(w, X, y)
    sg = (@parallel (+) for i = 1:m
        Obj.sg_lr(w, X, y, i)
    end) / m

    # assert
    success = (round(g, 6) == round(sg, 6)) # equal to the third decimal?
    if (success)
        println("Test of sg_lr SUCCEEDED.")
    else
        println("Test of sg_lr FAILED! Complete gradient at once = $g. Complete gradient using sg = $sg")
    end
    return success
end

tst_sg_lr()