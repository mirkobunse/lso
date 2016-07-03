# 
# Evaluation with Mnist data
# 
module Mnist

using Colors
using MAT
using Images
using ImageView

using GD
using LinReg
using LogReg
using Plotting


"""
    Show a row of the MNIST data matrix as image
"""
view_mnist(example) = ImageView.view(Images.grayim(convert( Images.Image{Gray}, reshape(example, (28,28)) )), xy=["y","x"])


"""
    gd_bt_mnist()

    Test GD.gd_bt(...) with Linear Regression on MNIST data.
"""
function gd_bt(maxiter=1000)
    # read data
    println("Reading mnist.mat...")
    file = MAT.matopen("data/mnist.mat")
    X = full(read(file, "X"))
    y = full(read(file, "y"))[:]

    # prepare objective function
    f(w) = LinReg.f(w, X, y)
    g(w) = LinReg.g(w, X, y)

    # tst
    w0 = zeros(784) # randn(784)
    @time inf = GD.gd(f, g, w0, maxiter=maxiter, printiter=100)
    w = inf[end, :w]
    iter = inf[end, :iter]
    opt = inf[end, :opt]

    view_mnist(w)

    println("Plotting...")
    Base.display(Plotting.plot_inf(inf))

    return nothing
end


end