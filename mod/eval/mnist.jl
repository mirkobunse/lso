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
#     img["spatialorder"] = ["y", "x"]
#     return img
# end
# 
# """
#     mnist_view(img)
#     
#     View an image obtained by mnist_img.
# """
# mnist_view(img) = ImageView.view(img) # xy=["y","x"])
# 
# """
#     mnist_saveview(filename, imgview)
#     
#     Save the image obtained by mnist_view(example)
# """
# mnist_saveimg(filename, img) = Images.save(filename, img)



function mnist_gd_bt(; maxiter=10000, maxtime=60, ϵ=1e-3)
    _mnist(Opt.gd, maxiter=maxiter, maxtime=maxtime, ϵ=ϵ, assumedgrad=false)
end

function mnist_sgd_sbt(; maxiter=100000, maxtime=60, batchsize=10, ϵ=0.0)
    _mnist(Opt.sgd, maxiter=maxiter, maxtime=maxtime, batchsize=batchsize, ϵ=ϵ, assumedgrad=false)
end

function mnist_svrg_sbt(; maxiter=10000, maxtime=60, batchsize=10, estimation=10, strategy=:avg, ϵ=1e-3)
    _mnist(Opt.svrg, maxiter=maxiter, maxtime=maxtime, batchsize=batchsize, estimation=estimation, ϵ=ϵ, assumedgrad=true)
end

function _mnist(opt::Function;
                batchsize=1, estimation=10, strategy=:last, maxiter=10000, maxtime=30, ϵ=1e-3, assumedgrad=true)

    srand(1337)

    println("Reading data...")
    X_train = readdlm("data/seven_vs_all/X_train.dlm")
    y_train = vec(readdlm("data/seven_vs_all/y_train.dlm"))
    X_test  = readdlm("data/seven_vs_all/X_test.dlm")
    y_test  = vec(readdlm("data/seven_vs_all/y_test.dlm"))
    println("Data consists of $(size(X_train)[1]) training and $(size(X_test)[1]) test examples.")

    # tst
    w0 = zeros(784) # rand(784)
    inf = LsoBase.new_inf()
    obj = Obj.logreg(X_train, y_train)
    try
        @time inf = opt(obj, w0, ϵ=ϵ, maxiter=maxiter, maxtime=maxtime,
                        batchsize=batchsize, estimation=estimation, strategy=strategy)
    catch e
        try
            @time inf = opt(obj, w0, ϵ=ϵ, maxiter=maxiter, maxtime=maxtime,
                            batchsize=batchsize)
        catch e
            @time inf = opt(obj, w0, ϵ=ϵ, maxiter=maxiter, maxtime=maxtime)
        end
    end
    w = inf[end, :w]
    iterrate = inf[end, :iter] / inf[end, :time]

    # acc
    acc_train = LsoBase.acc(y_train, Obj.logreg_predict(w, X_train))
    acc_test  = LsoBase.acc(y_test, Obj.logreg_predict(w, X_test))
    println(@sprintf "\n%20s: %8.4f" "Training set acc" acc_train)
    println(@sprintf   "%20s: %8.4f" "Test set acc"     acc_test)
    println(@sprintf   "%20s: %8.4f" "Iterations / sec" iterrate)

    # ask user for plot
    print("\nPlot progress? (y/N): ")
    if startswith(readline(STDIN), "y")
        plot = Plotting.plot_inf(inf, obj, assumedgrad)
        Plotting.display_plot(plot)

        optname = methods(opt).name
        outfile = "./mnist_$optname$batchsize.pdf"
        print("\nDraw to $outfile? (y/N): ")
        if startswith(readline(STDIN), "y")
            Plotting.draw_plot(plot, outfile)
        end

        # print("\nShow weight vector? (y/N): ")
        # if startswith(readline(STDIN), "y")
        #     mnist_view(mnist_img(w))
        # end
    end

    return nothing

end