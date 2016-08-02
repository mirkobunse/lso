using Colors
# using Images
# using ImageView

import Opt
import Obj
import Plotting

# """
#     mnist_img(row)
#     
#     Obtain the row of the MNIST data matrix as image.
# """
# function mnist_img(row)
#     img = Images.grayim(convert( Images.Image{Gray}, reshape(row, (28,28)) ))
#     img["spatialorder"] = ["x", "y"]
#     return img
# end

# """
#     mnist_view(img)
#     
#     View an image obtained by mnist_img.
# """
# mnist_view(img) = ImageView.view(img) # xy=["y","x"])

# """
#     mnist_saveview(filename, imgview)
#     
#     Save the image obtained by mnist_view(example)
# """
# mnist_saveimg(filename, img) = Images.save(filename, img)



function mnist_gd_bt(; maxiter=10000, maxtime=60, ϵ=1e-3)
    _mnist(Opt.gd, maxiter=maxiter, maxtime=maxtime, ϵ=ϵ)
end

function mnist_sgd_sbt(; maxiter=10000, maxtime=60, batchsize=1, ϵ=0.0)
    _mnist(Opt.sgd, maxiter=maxiter, maxtime=maxtime, batchsize=batchsize, storeiter=500, ϵ=ϵ)
end

function mnist_svrg_sbt(; maxiter=10000, maxtime=60, batchsize=1, estimation=10, strategy=:avg, ϵ=1e-30)
    _mnist(Opt.svrg, maxiter=maxiter, maxtime=maxtime, batchsize=batchsize, estimation=estimation, storeiter=500, ϵ=ϵ)
end

function _mnist(opt::Function; batchsize=1, estimation=10, strategy=:last, maxiter=10000, storeiter=5, maxtime=30, ϵ=1e-3)

    println("Reading data...")
    X_train = readdlm("data/X_train.dlm")
    y_train = vec(readdlm("data/y_train.dlm"))
    X_test  = readdlm("data/X_test.dlm")
    y_test  = vec(readdlm("data/y_test.dlm"))
    println("Data consists of $(size(X_train)[1]) training and $(size(X_test)[1]) test examples.")

    # tst
    w0 = zeros(784) # rand(784)
    inf = LsoBase.new_inf()
    obj = Obj.logreg(X_train, y_train)
    try
        @time inf = opt(obj, w0, ϵ=ϵ, maxiter=maxiter, storeiter=storeiter, maxtime=maxtime,
                        batchsize=batchsize, estimation=estimation, strategy=strategy)
    catch e
        try
            @time inf = opt(obj, w0, ϵ=ϵ, maxiter=maxiter, storeiter=storeiter, maxtime=maxtime,
                            batchsize=batchsize)
        catch e
            @time inf = opt(obj, w0, ϵ=ϵ, maxiter=maxiter, storeiter=storeiter, maxtime=maxtime)
        end
    end
    w = inf[end, :w]
    f = obj.f(w)
    println(@sprintf "\nf = %9.3e" f)

    # acc
    acc_train = LsoBase.acc(y_train, Obj.logreg_predict(w, X_train))
    acc_test  = LsoBase.acc(y_test, Obj.logreg_predict(w, X_test))
    println("\nTraining set accuracy: $acc_train")
    println("\nTest set accuracy: $acc_test")

    # ask user for plot
    print("\nPlot progress? (y/N): ")
    if startswith(readline(STDIN), "y")
        Plotting.display_inf(inf, obj)

        optname = methods(opt).name
        outfile = "./mnist_$optname$batchsize.pdf"
        print("\nDraw to $outfile? (y/N): ")
        if startswith(readline(STDIN), "y")
            Plotting.draw_inf(inf, obj, outfile)
        end

        # print("\nShow weight vector? (y/N): ")
        # if startswith(readline(STDIN), "y")
        #     mnist_view(mnist_img(w))
        # end
    end

    return nothing

end