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
mnist_view(img) = ImageView.view(Images.grayim(convert( Images.Image{Gray}, reshape(img, (28,28)) )), xy=["y","x"])

"""
    mnist_gd_bt()

    Test GD with BT on Linear Regression of MNIST data.
"""
function mnist_gd_bt(maxiter=1000)

    println("Reading data...")
    file = MAT.matopen("data/mnist.mat")
    X = full(read(file, "X"))
    y = full(read(file, "y"))[:]

    # tst
    w0 = zeros(784) # randn(784)
    @time inf = Opt.gd(Obj.linreg(X, y), w0, maxiter=maxiter, printiter=100)
    w = inf[end, :w]
    iter = inf[end, :iter]
    opt = inf[end, :opt]

    println("Plotting...")
    Plotting.display_inf(inf)
    # mnist_view(w)

    return nothing

end