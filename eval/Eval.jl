# 
# Evaluation with Mnist data
# 
module Eval

using Gadfly
using Colors
using MAT
using Images
using ImageView

import Opt
import Obj


"""
    plotinf(infarr [, ylabel])

    Plot development of optimality in IterInfo array.
"""
function plotinf(infarr, ylabel="Optimality ‖∇f(x)‖∞")
    opts = [info.opt for info in infarr]
    plot(x=1:length(opts), y=opts, Scale.y_log, Geom.line, Guide.xlabel("Iteration"), Guide.ylabel(ylabel))
end


"""
    gd_bt_rosenbrock([maxiter])

    Test Opt.gd_bt() with the Rosenbrock function
"""
function gd_bt_rosenbrock(maxiter=1000000)
    # init rosenbrock function
    f(w) = 100*(w[2]-w[1]^2)^2 + (1-w[1])^2
    function g(w)
        g2 = 200*(w[2]-w[1]^2)
        return [-2*(w[1]*g2 + (1-w[1])), g2]
    end
    # tst
    w = zeros(2)
    @time Opt.gd_bt(f, g, w, maxiter=maxiter, printiter=10000)
end


"""
    gd_bt_rand([; maxiter, n, m])

    Test Opt.gd_bt() with random data
"""
function gd_bt_rand(maxiter=1000; n=1000, m=10000)
    # init random data
    w_true = randn(n)
    X = randn(m,n)
    y = X*w_true
    f(w) = Obj.lr_f(w, X, y)
    g(w) = Obj.lr_g(w, X, y)

    # tst
    w0 = randn(n)
    @time infarr = Opt.gd_bt(f, g, w0, maxiter=maxiter)
    w = infarr[end].w
    iter = infarr[end].iter
    opt = infarr[end].opt

    # assert
    success = isapprox(w, w_true, atol=1e-4)
    if (success)
        println("tst_gd_bt SUCCEEDED: Found true w in $iter steps (Optimality $opt)")
    else
        println("tst_gd_bt FAILED: Did not find true w in $iter steps (Optimality $opt).\nw = $w\nw_true = $w_true")
    end

    return infarr
end


"""
    Show a row of the MNIST data matrix as image
"""
view_mnist(example) = ImageView.view(grayim(convert( Image{Gray}, reshape(example, (28,28)) )), xy=["y","x"])


"""
    gd_bt_mnist()

    Test Opt.gd_bt(...) with Logistic Regression on MNIST data.
"""
function gd_bt_mnist(maxiter=1000)
    # read data
    file = matopen("data/mnist.mat")
    X = full(read(file, "X"))
    y = full(read(file, "y"))[:]

    # prepare objective function
    f(w) = Obj.lr_f(w, X, y)
    g(w) = Obj.lr_g(w, X, y)

    # tst
    w0 = randn(784)
    @time infarr = Opt.gd_bt(f, g, w0, maxiter=maxiter)
    w = infarr[end].w
    iter = infarr[end].iter
    opt = infarr[end].opt

    view_mnist(w)
    return infarr
end


end