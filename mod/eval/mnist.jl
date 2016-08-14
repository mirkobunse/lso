using Colors

import Ls
import Ls.LineSearch
import GdOpt
import GdOpt.GdOptimizer
import Obj
import Plotting


# Quick test functions for GD.
mnist_gd() = mnist(GdOpt.gd(), Ls.bt())
mnist_gd_boost() = mnist_boost(GdOpt.gd(), Ls.bt())


# Quick test functions for SGD.
mnist_sgd(batchsize=1) = mnist(GdOpt.sgd(), Ls.sbt(), ϵ=0.0, batchsize=batchsize)
mnist_sgd_boost(batchsize=1) = mnist_boost(GdOpt.sgd(), Ls.sbt(), ϵ=0.0, batchsize=batchsize)


# Quick test functions for SVRG.
function mnist_svrg(; batchsize=10, estiter=10, strategy=:last)
    mnist(GdOpt.svrg(estiter, strategy), Ls.sbt(), batchsize=batchsize, assumedgrad=true)
end
function mnist_svrg_boost(; batchsize=10, estiter=10, strategy=:last)
    mnist_boost(GdOpt.svrg(estiter, strategy), Ls.sbt(), batchsize=batchsize, assumedgrad=true)
end


"""
    mnist(gdopt, ls, [, folder, seed; ϵ, maxtime, batchsize, assumedgrad])

    Evaluate optimizer on MNIST data.
"""
function mnist(gdopt::GdOptimizer, ls::LineSearch, folder::ASCIIString="seven_vs_all", seed::Int32=1337;
               ϵ::Float64=1e-3, maxtime::Float64=60.0, batchsize::Int32=-1, assumedgrad=false)

    srand(seed)
    X_train, y_train, X_test, y_test = _mnist_data(folder)  # read data

    # optimize
    inf, obj, w, acc_train, acc_test, iterrate = _mnist_opt(
            gdopt, ls, X_train, y_train, X_test, y_test,
            ϵ, maxtime, batchsize
    )

    # ask user for plot
    outfile = "./results/$folder/$(gdopt.name)$((batchsize>0)?batchsize:"")_$(ls.name).pdf"
    print("\nDraw to $outfile? (y/N): ")
    if startswith(readline(STDIN), "y")
        println("Plotting...")
        plot = Plotting.plot_inf(inf, obj, assumedgrad)
        println("Drawing to $outfile...")
        Plotting.draw_plot(plot, outfile)
    end
    
    return nothing

end



function mnist_boost(gdopt::GdOptimizer, ls::LineSearch, folder::ASCIIString="seven_vs_all", seed::Int32=1337;
                     ϵ::Float64=1e-3, maxtime::Float64=60.0, batchsize::Int32=-1, assumedgrad=false, frac1=.5)
    
    srand(seed)
    X_train, y_train, X_test, y_test = _mnist_data(folder)  # read data

    ################
    # ITERATION 1
    ################

    # random subset of relative size frac1
    train1 = randperm(length(y_train))[1:convert(Int32, floor(frac1*length(y_train)))]
    X_train1 = X_train[train1,:]
    y_train1 = y_train[train1]

    # optimize
    inf1, w1, acc_train1, acc_test1, iterrate1 = _mnist_ensemble_opt(
            gdopt, ls, X_train1, y_train1, X_test, y_test,
            ϵ, maxtime, batchsize
    )

    ################
    # ITERATION 2
    ################

    # find subset that is classified with 50% acc
    y_pred1    = Obj.logreg_predict(w1, X_train)
    correct1   = shuffle(find(y_train .== y_pred1))
    false1     = shuffle(find(y_train .!= y_pred1))
    train2size = min(length(correct1), length(false1))
    X_train2   = getindex(X_train, vcat(correct1[1:train2size], false1[1:train2size]), :)
    y_train2   = getindex(y_train, vcat(correct1[1:train2size], false1[1:train2size]))

    # optimize
    inf2, w2, acc_train2, acc_test2, iterrate2 = _mnist_ensemble_opt(
            gdopt, ls, X_train2, y_train2, X_test, y_test,
            ϵ, maxtime, batchsize
    )

    ################
    # ITERATION 3
    ################

    # find subset for which predictors decide differently
    y_pred2 = Obj.logreg_predict(w2, X_train)
    X_train3 = X_train[(y_pred1 .!= y_pred2), :]
    y_train3 = y_train[(y_pred1 .!= y_pred2)]

    # optimize
    inf3, w3, acc_train3, acc_test3, iterrate3 = _mnist_ensemble_opt(
            gdopt, ls, X_train3, y_train3, X_test, y_test,
            ϵ, maxtime, batchsize
    )

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
    numex     = size(unique(vcat(X_train1, X_train2, X_train3), 1))[1]
    iterrate  = mean([iterrate1, iterrate2, iterrate3])

    # conclusion
    headline = @sprintf "\n\n%16s | %10s | %10s | %10s | %10s | %10s" "Decision" "Train Acc" "Test Acc" "Time (s)" "Train Size" "Iter / sec"
    println(headline, "\n", repeat("-", length(headline)))
    resline = (c::ASCIIString, tr::Float64, te::Float64, ti::Float64, ex::Int32, it::Float64) ->
    println(@sprintf "%16s | %10.3f | %10.3f | %10.3f | %10d | %10.3f" c tr te ti ex it)
    resline("1st Classifier", acc_train1, acc_test1, inf1[end, :time], size(X_train1)[1], iterrate1)
    resline("2nd Classifier", acc_train2, acc_test2, inf2[end, :time], size(X_train2)[1], iterrate2)
    resline("3rd Classifier", acc_train3, acc_test3, inf3[end, :time], size(X_train3)[1], iterrate3)
    println(repeat("-", length(headline)))
    resline("Majority Vote",  acc_train,  acc_test,  timesum,          numex,             iterrate)
    println(repeat("-", length(headline)), "\n")

    # ask user for plot
    outfile = "./results/$folder/boost_$(gdopt.name)$((batchsize>0)?batchsize:"")_$(ls.name).pdf"
    print("\nDraw to $outfile? (y/N): ")
    if startswith(readline(STDIN), "y")
        println("Plotting...")
        inf2[:time] += inf1[end, :time]
        inf2[:iter] += inf1[end, :iter]
        inf3[:time] += inf2[end, :time]
        inf3[:iter] += inf2[end, :iter]
        inf = vcat(inf1, inf2, inf3)
        # vlines = [inf1[end, :time], inf2[end, :time], inf3[end, :time]]
        plot = Plotting.plot_inf(inf, Obj.logreg(X_train, y_train), assumedgrad)
        println("Drawing to $outfile...")
        Plotting.draw_plot(plot, outfile)
    end
    
    return nothing

end


function _mnist_data(folder::ASCIIString)
    println("Reading data from ./data/$folder...")
    X_train = readdlm("./data/$folder/X_train.dlm")
    y_train = vec(readdlm("./data/$folder/y_train.dlm"))
    X_test  = readdlm("./data/$folder/X_test.dlm")
    y_test  = vec(readdlm("./data/$folder/y_test.dlm"))
    return X_train, y_train, X_test, y_test
end


function _mnist_opt(gdopt::GdOptimizer, ls::LineSearch, X_train, y_train, X_test, y_test,
                    ϵ::Float64, maxtime::Float64, batchsize::Int32)
    
    println("\nNow considering $(size(X_train)[1]) training examples...")
    
    # optimize
    obj = Obj.logreg(X_train, y_train)
    @time inf = GdOpt.opt(gdopt, ls, obj, zeros(784),
                          ϵ=ϵ, maxtime=maxtime, batchsize=batchsize)
    w = inf[end, :w]
    iterrate = inf[end, :iter] / inf[end, :time]

    # acc
    acc_train = LsoBase.acc(y_train, Obj.logreg_predict(w, X_train))
    acc_test  = LsoBase.acc(y_test, Obj.logreg_predict(w, X_test))
    println("\n  Training set acc: $acc_train")
    println(  "      Test set acc: $acc_test")
    println(  "  Iterations / sec: $iterrate")

    return inf, obj, w, acc_train, acc_test, iterrate

end