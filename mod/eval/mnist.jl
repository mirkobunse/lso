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

    Test GD with BT on Logistic Regression of MNIST data.
"""
function mnist_gd_bt(maxiter=1000)

    println("Reading data...")
    file = MAT.matopen("data/mnist.mat")
    X = full(read(file, "X"))
    y = full(read(file, "y"))[:]
    println(size(X))

    # tst
    w0 = zeros(784) # rand(784)
    @time inf = Opt.gd(Obj.logreg(X, y), w0, maxiter=maxiter, printiter=5)
    w = inf[end, :w]
    iter = inf[end, :iter]
    opt = inf[end, :opt]

    # acc
    acc = LsoBase.acc(y, Obj.logreg_predict(w, X))
    println("\nTraining set accuracy: $acc")

    # ask user for plot
    print("\nPlot progress? (y/N): ")
    if startswith(readline(STDIN), "y")
        mnist_view(w)
        Plotting.display_inf(inf)
    end

    return nothing

end