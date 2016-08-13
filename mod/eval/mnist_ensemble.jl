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



function mnist_gd_bt_ensemble(; maxiter=10000, maxtime=60, ϵ=1e-3)
    _mnist_ensemble(Opt.gd, maxiter=maxiter, maxtime=maxtime, ϵ=ϵ, assumedgrad=false)
end

function mnist_sgd_sbt_ensemble(; maxiter=100000, maxtime=60, batchsize=1, ϵ=0.0)
    _mnist_ensemble(Opt.sgd, maxiter=maxiter, maxtime=maxtime, batchsize=batchsize, storeiter=500, ϵ=ϵ, assumedgrad=false)
end

function mnist_svrg_sbt_ensemble(; maxiter=10000, maxtime=60, batchsize=1, estimation=100, strategy=:avg, ϵ=1e-30)
    _mnist_ensemble(Opt.svrg, maxiter=maxiter, maxtime=maxtime, batchsize=batchsize, estimation=estimation, storeiter=100, ϵ=ϵ, assumedgrad=true)
end

function _mnist_ensemble(opt::Function;
                batchsize=1, estimation=10, strategy=:last, maxiter=10000, storeiter=5, maxtime=30, ϵ=1e-3, assumedgrad=true,
                frac1=.5)

    srand(1337)

    ################
    # DATA
    ################

    println("Reading data...")
    X_train = readdlm("data/seven_vs_all/X_train.dlm")
    y_train = vec(readdlm("data/seven_vs_all/y_train.dlm"))
    X_test  = readdlm("data/seven_vs_all/X_test.dlm")
    y_test  = vec(readdlm("data/seven_vs_all/y_test.dlm"))
    println("Data consists of $(size(X_train)[1]) training and $(size(X_test)[1]) test examples.")



    ################
    # ITERATION 1
    ################

    # random subset of relative size frac1
    train1 = randperm(length(y_train))[1:convert(Int32, floor(frac1*length(y_train)))]
    X_train1 = X_train[train1,:]
    y_train1 = y_train[train1]
    println("\n1st iteration considers $(size(X_train1)[1]) training examples.")

    # optimize
    w0 = zeros(784) # rand(784)
    inf1 = LsoBase.new_inf()
    obj1 = Obj.logreg(X_train1, y_train1)
    try
        @time inf1 = opt(obj1, w0, ϵ=ϵ, maxiter=maxiter, storeiter=storeiter, maxtime=maxtime,
                        batchsize=batchsize, estimation=estimation, strategy=strategy)
    catch e
        try
            @time inf1 = opt(obj1, w0, ϵ=ϵ, maxiter=maxiter, storeiter=storeiter, maxtime=maxtime,
                            batchsize=batchsize)
        catch e
            @time inf1 = opt(obj1, w0, ϵ=ϵ, maxiter=maxiter, storeiter=storeiter, maxtime=maxtime)
        end
    end
    w1 = inf1[end, :w]
    iterrate1 = inf1[end, :iter] / inf1[end, :time]

    # acc
    acc_train1 = LsoBase.acc(y_train1, Obj.logreg_predict(w1, X_train1))
    acc_test1  = LsoBase.acc(y_test, Obj.logreg_predict(w1, X_test))
    println(@sprintf "\n%20s: %8.4f" "Training set acc" acc_train1)
    println(@sprintf   "%20s: %8.4f" "Test set acc"     acc_test1)
    println(@sprintf   "%20s: %8.4f" "Iterations / sec" iterrate1)



    ################
    # ITERATION 2
    ################

    # find subset that is classified with 50% acc
    y_pred1 = Obj.logreg_predict(w1, X_train)
    correct1 = shuffle(find(y_train .== y_pred1))
    false1   = shuffle(find(y_train .!= y_pred1))
    train2size = min(length(correct1), length(false1))
    train2indices = vcat(correct1[1:train2size], false1[1:train2size])
    X_train2 = getindex(X_train, train2indices, :)
    y_train2 = getindex(y_train, train2indices)
    println("\n2nd iteration will consider $(size(X_train2)[1]) training examples.")

    # optimize
    inf2 = LsoBase.new_inf()
    obj2 = Obj.logreg(X_train2, y_train2)
    try
        @time inf2 = opt(obj2, w0, ϵ=ϵ, maxiter=maxiter, storeiter=storeiter, maxtime=maxtime,
                        batchsize=batchsize, estimation=estimation, strategy=strategy)
    catch e
        try
            @time inf2 = opt(obj2, w0, ϵ=ϵ, maxiter=maxiter, storeiter=storeiter, maxtime=maxtime,
                            batchsize=batchsize)
        catch e
            @time inf2 = opt(obj2, w0, ϵ=ϵ, maxiter=maxiter, storeiter=storeiter, maxtime=maxtime)
        end
    end
    w2 = inf2[end, :w]
    iterrate2 = inf2[end, :iter] / inf2[end, :time]

    # acc
    acc_train2 = LsoBase.acc(y_train2, Obj.logreg_predict(w2, X_train2))
    acc_test2  = LsoBase.acc(y_test, Obj.logreg_predict(w2, X_test))
    println(@sprintf "\n%20s: %8.4f" "Training set acc" acc_train2)
    println(@sprintf   "%20s: %8.4f" "Test set acc"     acc_test2)
    println(@sprintf   "%20s: %8.4f" "Iterations / sec" iterrate2)



    ################
    # ITERATION 3
    ################

    # find subset for which predictors decide differently
    y_pred2 = Obj.logreg_predict(w2, X_train)
    different = (y_pred1 .!= y_pred2)
    X_train3 = X_train[different, :]
    y_train3 = y_train[different]
    println("\n3rd iteration will consider $(size(X_train3)[1]) training examples.")

    # optimize
    inf3 = LsoBase.new_inf()
    obj3 = Obj.logreg(X_train3, y_train3)
    try
        @time inf3 = opt(obj3, w0, ϵ=ϵ, maxiter=maxiter, storeiter=storeiter, maxtime=maxtime,
                        batchsize=batchsize, estimation=estimation, strategy=strategy)
    catch e
        try
            @time inf3 = opt(obj3, w0, ϵ=ϵ, maxiter=maxiter, storeiter=storeiter, maxtime=maxtime,
                            batchsize=batchsize)
        catch e
            @time inf3 = opt(obj3, w0, ϵ=ϵ, maxiter=maxiter, storeiter=storeiter, maxtime=maxtime)
        end
    end
    w3 = inf3[end, :w]
    iterrate3 = inf3[end, :iter] / inf3[end, :time]

    # acc
    acc_train3 = LsoBase.acc(y_train3, Obj.logreg_predict(w3, X_train3))
    acc_test3  = LsoBase.acc(y_test, Obj.logreg_predict(w3, X_test))
    println(@sprintf "\n%20s: %8.4f" "Training set acc" acc_train3)
    println(@sprintf   "%20s: %8.4f" "Test set acc"     acc_test3)
    println(@sprintf   "%20s: %8.4f" "Iterations / sec" iterrate3)



    ################
    # MAJORITY VOTE
    ################

    # training prediction
    y_trainpred1 = Obj.logreg_predict(w1, X_train)
    y_trainpred2 = Obj.logreg_predict(w2, X_train)
    y_trainpred3 = Obj.logreg_predict(w3, X_train)
    y_trainpred  = vec(sign(sum([y_trainpred1 y_trainpred2 y_trainpred3], 2)))

    # test prediction
    y_testpred1  = Obj.logreg_predict(w1, X_test)
    y_testpred2  = Obj.logreg_predict(w2, X_test)
    y_testpred3  = Obj.logreg_predict(w3, X_test)
    y_testpred   = vec(sign(sum([y_testpred1 y_testpred2 y_testpred3], 2)))

    # acc
    acc_train = LsoBase.acc(y_train, y_trainpred)
    acc_test  = LsoBase.acc(y_test, y_testpred)
    timesum   = sum([inf1[end, :time], inf2[end, :time], inf3[end, :time]])
    numexamples = size(unique(vcat(X_train1, X_train2, X_train3), 1))[1]
    meaniterrate = mean([iterrate1, iterrate2, iterrate3])
    println(@sprintf "\n%30s: %8.4f" "Overall Training set acc" acc_train)
    println(@sprintf   "%30s: %8.4f" "Overall Test set acc"     acc_test)
    println(@sprintf   "%30s: %8d"   "Overall # examples"       numexamples)

    # conclusion
    headline = @sprintf "\n\n%16s | %10s | %10s | %10s | %10s | %10s" "Decision" "Train Acc" "Test Acc" "Time (s)" "Train Size" "Iter / sec"
    println(headline, "\n", repeat("-", length(headline)))
    println(@sprintf "%16s | %10.3f | %10.3f | %10.3f | %10d | %10.3f" "1st Classifier" acc_train1 acc_test1 inf1[end, :time] size(X_train1)[1] iterrate1)
    println(@sprintf "%16s | %10.3f | %10.3f | %10.3f | %10d | %10.3f" "2nd Classifier" acc_train2 acc_test2 inf2[end, :time] size(X_train2)[1] iterrate2)
    println(@sprintf "%16s | %10.3f | %10.3f | %10.3f | %10d | %10.3f" "3rd Classifier" acc_train3 acc_test3 inf3[end, :time] size(X_train3)[1] iterrate3)
    println(@sprintf "%16s | %10.3f | %10.3f | %10.3f | %10d | %10.3f" "Majority Vote"  acc_train  acc_test  timesum          numexamples       meaniterrate)
    println("")



    ################
    # PLOTTING
    ################
    print("\nPlot combined progress? (y/N): ")
    if startswith(readline(STDIN), "y")
        inf2[:time] += inf1[end, :time]
        inf2[:iter] += inf1[end, :iter]
        inf3[:time] += inf2[end, :time]
        inf3[:iter] += inf2[end, :iter]
        inf = vcat(inf1, inf2, inf3)
        vlines = [inf1[end, :time], inf2[end, :time], inf3[end, :time]]
        plot = Plotting.plot_inf(inf, Obj.logreg(X_train, y_train), assumedgrad, vlines=vlines)
        Plotting.display_plot(plot)

        optname = methods(opt).name
        outfile = "./mnist_$optname$(batchsize)_ensemble.pdf"
        print("\nDraw to $outfile? (y/N): ")
        if startswith(readline(STDIN), "y")
            Plotting.draw_plot(plot, outfile)
        end
    end


    return nothing


end