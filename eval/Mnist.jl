# 
# Evaluation with Mnist data
# 
module Mnist

using Gadfly
using Colors
import MAT
import Images
import ImageView

import GD
import LinReg


"""
    Show a row of the MNIST data matrix as image
"""
view_mnist(example) = ImageView.view(Images.grayim(convert( Images.Image{Gray}, reshape(example, (28,28)) )), xy=["y","x"])


"""
    gd_bt_mnist()

    Test GD.gd_bt(...) with Logistic Regression on MNIST data.
"""
function gd_bt_mnist(maxiter=1000)
    # read data
    file = matopen("data/mnist.mat")
    X = full(read(file, "X"))
    y = full(read(file, "y"))[:]

    # prepare objective function
    f(w) = LinReg.f(w, X, y)
    g(w) = LinReg.g(w, X, y)

    # tst
    w0 = randn(784)
    @time infarr = GD.gd_bt(f, g, w0, maxiter=maxiter)
    w = infarr[end].w
    iter = infarr[end].iter
    opt = infarr[end].opt

    view_mnist(w)
    return infarr
end


end