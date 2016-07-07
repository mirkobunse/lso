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
function mnist_gd_bt(; maxiter=10000)
    _mnist(Opt.gd, maxiter=maxiter)
end

function mnist_sgd_bt(; maxiter=10000, batchSize=1)
    _mnist(Opt.sgd, maxiter=maxiter, batchSize=batchSize, timeiter=10)
end

function _mnist(opt::Function; batchSize=1, maxiter=10000, timeiter=5)

    println("Reading data...")
    file = MAT.matopen("data/mnist.mat")
    X = full(read(file, "X"))
    y = full(read(file, "y"))[:]
    println("Data set contains $(size(X)[1]) examples of dimension $(size(X)[2]).")

    # split into test and training set (shuffled)
    perm = randperm(length(y))
    train = perm[1:convert(Int32, floor(length(perm)*2/3))]
    test  = perm[convert(Int32, ceil(length(perm)*2/3)):end]
    X_train = X[train, :]
    X_test  = X[test, :]
    y_train = y[train]
    y_test  = y[test]
    println("Shuffled into training set ($(size(X_train)[1]) examples) and test set ($(size(X_test)[1]) examples).")

    # tst
    w0 = zeros(784) # rand(784)
    inf = LsoBase.new_inf()
    try
        @time inf = opt(Obj.logreg(X_train, y_train), w0, ϵ=1e-3, maxiter=maxiter, timeiter=timeiter, batchSize=batchSize)
    catch e
        @time inf = opt(Obj.logreg(X_train, y_train), w0, ϵ=1e-3, maxiter=maxiter, timeiter=timeiter)
    end
    w = inf[end, :w]
    iter = inf[end, :iter]
    opt = inf[end, :opt]

    # acc
    acc_train = LsoBase.acc(y_train, Obj.logreg_predict(w, X_train))
    acc_test  = LsoBase.acc(y_test, Obj.logreg_predict(w, X_test))
    println("\nTraining set accuracy: $acc_train")
    println("\nTest set accuracy: $acc_test")

    # ask user for plot
    print("\nPlot progress? (y/N): ")
    if startswith(readline(STDIN), "y")
        mnist_view(w)
        Plotting.display_inf(inf)
    end

    return nothing

end