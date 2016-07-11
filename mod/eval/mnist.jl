using Colors
using MAT
using Images
using ImageView

import Opt
import Obj
import Plotting

"""
    mnist_img(row)

    Obtain the row of the MNIST data matrix as image.
"""
function mnist_img(row)
    img = Images.grayim(convert( Images.Image{Gray}, reshape(row, (28,28)) ))
    img["spatialorder"] = ["x", "y"]
    return img
end

"""
    mnist_view(img)

    View an image obtained by mnist_img.
"""
mnist_view(img) = ImageView.view(img) # xy=["y","x"])

"""
    mnist_saveview(filename, imgview)

    Save the image obtained by mnist_view(example)
"""
mnist_saveimg(filename, img) = Images.save(filename, img)

"""
    mnist_readdata()

    Return the data as (X, y) tuple, where X is the example matrix and y is the labels.
"""
function mnist_readdata()
    println("Reading data...")
    file = MAT.matopen("data/mnist.mat")
    X = full(read(file, "X"))
    y = full(read(file, "y"))[:]
    close(file)
    return X, y
end



function mnist_gd_bt(; maxiter=10000, ϵ=1e-3)
    _mnist(Opt.gd, maxiter=maxiter, ϵ=ϵ)
end

function mnist_sgd_bt(; maxiter=10000, batchSize=1, ϵ=1e-4)
    _mnist(Opt.sgd, maxiter=maxiter, batchSize=batchSize, ϵ=ϵ)
end

function _mnist(opt::Function; batchSize=1, maxiter=10000, timeiter=5, ϵ=1e-3)

    X, y = mnist_readdata()
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
        @time inf = opt(Obj.logreg(X_train, y_train), w0, ϵ=ϵ, maxiter=maxiter, timeiter=timeiter, batchSize=batchSize)
    catch e
        @time inf = opt(Obj.logreg(X_train, y_train), w0, ϵ=ϵ, maxiter=maxiter, timeiter=timeiter)
    end
    w = inf[end, :w]

    # acc
    acc_train = LsoBase.acc(y_train, Obj.logreg_predict(w, X_train))
    acc_test  = LsoBase.acc(y_test, Obj.logreg_predict(w, X_test))
    println("\nTraining set accuracy: $acc_train")
    println("\nTest set accuracy: $acc_test")

    # ask user for plot
    print("\nPlot progress? (y/N): ")
    if startswith(readline(STDIN), "y")
        Plotting.display_inf(inf)

        optname = methods(opt).name
        outfile = "./mnist_$optname$batchSize.pdf"
        print("\nDraw to $outfile? (y/N): ")
        if startswith(readline(STDIN), "y")
            Plotting.draw_inf(inf, outfile)
        end

        print("\nShow weight vector? (y/N): ")
        if startswith(readline(STDIN), "y")
            mnist_view(mnist_img(w))
        end
    end

    return nothing

end