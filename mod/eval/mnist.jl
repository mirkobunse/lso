using Colors
# using Images
# using ImageView

import Ls
import Ls.LineSearch
import GdOpt
import GdOpt.GdOptimizer
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



function mnist_gd_bt(; ϵ=1e-3, maxtime=60.0)
    _mnist(GdOpt.gd(), Ls.bt, ϵ, maxtime)
end

function mnist_sgd_sbt(; ϵ=0.0, maxtime=60.0, batchsize=10)
    _mnist(GdOpt.sgd(), Ls.sbt, ϵ, maxtime, batchsize)
end

function mnist_svrg_sbt(; ϵ=1e-3, maxtime=60.0, batchsize=10, estiter=10, strategy=:last)
    _mnist(GdOpt.svrg(estiter, strategy), Ls.sbt, ϵ, maxtime, batchsize, assumedgrad=true)
end

function _mnist(optimizer::GdOptimizer, ls::Function, ϵ::Float64, maxtime::Float64, batchsize::Int32=-1;
                assumedgrad=false)

    srand(1337)

    println("Reading data...")
    X_train = readdlm("data/seven_vs_all/X_train.dlm")
    y_train = vec(readdlm("data/seven_vs_all/y_train.dlm"))
    X_test  = readdlm("data/seven_vs_all/X_test.dlm")
    y_test  = vec(readdlm("data/seven_vs_all/y_test.dlm"))
    println("Data consists of $(size(X_train)[1]) training and $(size(X_test)[1]) test examples.")

    # tst
    obj = Obj.logreg(X_train, y_train)
    @time inf = GdOpt.opt(optimizer, ls(obj), obj, zeros(784),
                          ϵ=ϵ, maxtime=maxtime, batchsize=batchsize)
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

        optname = optimizer.name
        lsname  = methods(ls).name
        outfile = "./mnist_$optname$(batchsize)_$lsname.pdf"
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