using Colors
using MAT
using Images
using ImageView

import Opt
import Obj
import Plotting

"""
    Show a row of the MNIST data matrix as image
"""
mnist_view(example) = ImageView.view(Images.grayim(convert( Images.Image{Gray}, reshape(example, (28,28)) )), xy=["y","x"])

"""
    mnist_gd_bt()

    Test GD with BT on Linear Regression of MNIST data.
"""
function mnist_gd_bt(maxiter=1000)
    # read data
    println("Reading mnist.mat...")
    file = MAT.matopen("data/mnist.mat")
    X = full(read(file, "X"))
    y = full(read(file, "y"))[:]

    # prepare objective function
    f(w) = Obj.f_linreg(w, X, y)
    g(w) = Obj.g_linreg(w, X, y)

    # tst
    w0 = zeros(784) # randn(784)
    @time inf = Opt.gd(f, g, w0, maxiter=maxiter, printiter=100)
    w = inf[end, :w]
    iter = inf[end, :iter]
    opt = inf[end, :opt]

    # mnist_view(w)

    println("Plotting...")
    Base.display(Plotting.plot_inf(inf))

    return nothing
end