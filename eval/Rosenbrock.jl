# 
module Rosenbrock

using LsoBase
using GD
using LinReg


"""
    gd_bt_rosenbrock([maxiter])

    Test GD.gd_bt() with the Rosenbrock function
"""
function gd_bt(maxiter::Int32=convert(Int32, 1e5))
    # init rosenbrock function
    f(w) = 100*(w[2]-w[1]^2)^2 + (1-w[1])^2
    function g(w)
        g2 = 200*(w[2]-w[1]^2)
        return [-2*(w[1]*g2 + (1-w[1])), g2]
    end
    # tst
    w = zeros(2)
    inf = @time GD.gd(f, g, w, maxiter=maxiter, printiter=10000)

    return nothing # inf
end


end