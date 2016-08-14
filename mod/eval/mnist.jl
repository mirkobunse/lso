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
mnist_svrg(; batchsize=10, estiter=10, strategy=:last, ϵ=0.0) =
    mnist(GdOpt.svrg(estiter, strategy), Ls.sbt(), batchsize=batchsize, ϵ=ϵ, assumedgrad=true)
mnist_svrg_boost(; batchsize=10, estiter=10, strategy=:last) =
    mnist_boost(GdOpt.svrg(estiter, strategy), Ls.sbt(), batchsize=batchsize, ϵ=ϵ, assumedgrad=true)


"""
    mnist_massive()

    Massive evaluation using all of the optimizers with different parameters and seeds.
    This may run up to 12hrs.
"""
function mnist_massive(folder::ASCIIString="seven_vs_all")

    # collect experiments
    seeds  = 1337:(1337+9)  # try 10 seeds
    gdopts = [GdOpt.gd(), GdOpt.sgd(),
              GdOpt.svrg(5, :last),  GdOpt.svrg(5, :avg),
              GdOpt.svrg(10, :last), GdOpt.svrg(10, :avg),
              GdOpt.svrg(25, :last), GdOpt.svrg(25, :avg)]
    batchsizes = [1, 10, 25, 100]

    # let user confirm his madness
    numexperiments = length(seeds) * (length(gdopts)+1) * length(batchsizes) * 4
    print("\nThis will run $numexperiments experiments, ",
          "that may last up to $(numexperiments/120)hrs and use up to $(numexperiments*6)MB of storage.\nPlease confirm: y/N? ")
    if !startswith(readline(STDIN), "y")
        return nothing
    end

    X_train, y_train, X_test, y_test = _mnist_data(folder)  # read data

    # iteration printer
    expiter = 0
    printiter() = println("\n", repeat("-", 25),
                          " Finished iteration $(expiter += 1), ",
                          "still $((numexperiments-expiter)) ",
                          "($(floor((numexperiments-expiter)/12)/10)hrs) to go ",
                          repeat("-", 25), "\n")

    # go for it
    for seed in seeds
        df = LsoBase.new_acc()
        for gdopt in gdopts

            if gdopt.name == "gd"
                try
                    df = vcat(df, mnist(gdopt, Ls.bt(), folder, seed,
                                        X_train=X_train, y_train=y_train,
                                        X_test=X_test, y_test=y_test))
                catch e
                    warn(e)
                    writedlm("./results/$folder/$(seed)_$(gdopt.name)$((batchsize>0)?batchsize:"")_$(Ls.bt().name).dlm", e)
                end
                printiter()

                try
                    df = vcat(df, mnist_boost(gdopt, Ls.bt(), folder, seed,
                                              X_train=X_train, y_train=y_train,
                                              X_test=X_test, y_test=y_test))
                catch e
                    warn(e)
                    writedlm("./results/$folder/boost_$(seed)_$(gdopt.name)$((batchsize>0)?batchsize:"")_$(Ls.bt().name).dlm", e)
                end
                printiter()
            end

            for batchsize in batchsizes
                ϵ=1e-3
                if (gdopt.name == "sgd" && batchsize != 100) || (gdopt.name == "svrg" && batchsize == 1)
                    ϵ=0.0
                end

                try
                    df = vcat(df, mnist(gdopt, Ls.sbt(), folder, seed, batchsize=batchsize, ϵ=ϵ,
                                        X_train=X_train, y_train=y_train,
                                        X_test=X_test, y_test=y_test))
                catch e
                    warn(e)
                    writedlm("./results/$folder/$(seed)_$(gdopt.name)$((batchsize>0)?batchsize:"")_$(Ls.sbt().name).dlm", e)
                end
                printiter()

                try
                    df = vcat(df, mnist_boost(gdopt, Ls.sbt(), folder, seed, batchsize=batchsize, ϵ=ϵ,
                                              X_train=X_train, y_train=y_train,
                                              X_test=X_test, y_test=y_test))
                catch e
                    warn(e)
                    writedlm("./results/$folder/boost_$(seed)_$(gdopt.name)$((batchsize>0)?batchsize:"")_$(Ls.sbt().name).dlm", e)
                end
                printiter()
            end

        end
        LsoBase.store(df, "./results/$folder/$(seed).csv")
    end

    println("\nDone!\n")

end

"""
    mnist(gdopt, ls, [, folder, seed; ϵ, maxtime, batchsize, assumedgrad, storage])

    Evaluate optimizer on MNIST data with Logistic Regression.
"""
function mnist(gdopt::GdOptimizer, ls::LineSearch, folder::ASCIIString="seven_vs_all", seed::Int32=1337;
               ϵ::Float64=1e-3, maxtime::Float64=30.0, batchsize::Int32=-1, assumedgrad=false, storage=true,
               X_train::Union{Void,Array{Float64,2}}=nothing, y_train::Union{Void,Array{Float64,1}}=nothing,
               X_test::Union{Void,Array{Float64,2}}=nothing,  y_test::Union{Void,Array{Float64,1}}=nothing)

    println("\nRunning Eval.mnist with seed $seed on $(gdopt.name) with $(ls.name) and batchsize $batchsize...")
    srand(seed)
    X_train, y_train, X_test, y_test = _mnist_data(folder, X_train, y_train, X_test, y_test)  # read data

    # optimize
    inf, obj, w, acc_train, acc_test, time, iterrate = _mnist_opt(
            gdopt, ls, X_train, y_train, X_test, y_test,
            ϵ, maxtime, batchsize
    )

    # plotting and storage
    if storage
        filename = "./results/$folder/$(seed)_$(gdopt.name)$((batchsize>0)?batchsize:"")_$(ls.name)"
        LsoBase.store(inf, filename * ".csv")
        Plotting.draw_plot(Plotting.plot_inf(inf, assumedgrad), filename * ".pdf")
    end
    
    println("")
    df = LsoBase.new_acc()
    LsoBase.push_acc!(df, gdopt.name, ls.name, batchsize, seed, folder, "LogReg",
                      size(X_train)[1], acc_train, acc_test, time, iterrate)
    return df

end


"""
    mnist(gdopt, ls, [, folder, seed; ϵ, maxtime, batchsize, assumedgrad, storage])

    Evaluate optimizer on MNIST data with historic boosting.
"""
function mnist_boost(gdopt::GdOptimizer, ls::LineSearch, folder::ASCIIString="seven_vs_all", seed::Int32=1337;
                     ϵ::Float64=1e-3, maxtime::Float64=30.0, batchsize::Int32=-1, assumedgrad=false, frac1=.5, storage=true,
                     X_train::Union{Void,Array{Float64,2}}=nothing, y_train::Union{Void,Array{Float64,1}}=nothing,
                     X_test::Union{Void,Array{Float64,2}}=nothing,  y_test::Union{Void,Array{Float64,1}}=nothing)
    
    println("\nRunning Eval.mnist_boost with seed $seed on $(gdopt.name) with $(ls.name) and batchsize $batchsize...")
    srand(seed)
    X_train, y_train, X_test, y_test = _mnist_data(folder, X_train, y_train, X_test, y_test)  # read data

    ################
    # ITERATION 1
    ################

    # random subset of relative size frac1
    train1 = randperm(length(y_train))[1:convert(Int32, floor(frac1*length(y_train)))]
    X_train1 = X_train[train1,:]
    y_train1 = y_train[train1]

    # optimize
    inf1, obj1, w1, acc_train1, acc_test1, time1, iterrate1 = _mnist_opt(
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
    inf2, obj2, w2, acc_train2, acc_test2, time2, iterrate2 = _mnist_opt(
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
    inf3, obj3, w3, acc_train3, acc_test3, time3, iterrate3 = _mnist_opt(
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
    timesum   = sum([time1, time2, time3])
    numex     = size(unique(vcat(X_train1, X_train2, X_train3), 1))[1]
    iterrate  = mean([iterrate1, iterrate2, iterrate3])

    # conclusion
    df = LsoBase.new_acc()
    LsoBase.push_acc!(df, gdopt.name, ls.name, batchsize, seed, folder, "LogReg B1",
                      size(X_train1)[1], acc_train1, acc_test1, time1,   iterrate1)
    LsoBase.push_acc!(df, gdopt.name, ls.name, batchsize, seed, folder, "LogReg B2",
                      size(X_train2)[1], acc_train2, acc_test2, time2,   iterrate2)
    LsoBase.push_acc!(df, gdopt.name, ls.name, batchsize, seed, folder, "LogReg B3",
                      size(X_train3)[1], acc_train3, acc_test3, time3,   iterrate3)
    LsoBase.push_acc!(df, gdopt.name, ls.name, batchsize, seed, folder, "Boosting B1-3",
                      numex,             acc_train,  acc_test,  timesum, iterrate)
    println(string(df), "\n")

    # concatenate inf
    inf2[:time] += time1
    inf2[:iter] += inf1[end, :iter]
    inf3[:time] += time2 + time1
    inf3[:iter] += inf2[end, :iter]
    inf = vcat(inf1, inf2, inf3)

    # plotting and storage
    if storage
        filename = "./results/$folder/boost_$(seed)_$(gdopt.name)$((batchsize>0)?batchsize:"")_$(ls.name)"
        LsoBase.store(inf, filename * ".csv")
        Plotting.draw_plot(Plotting.plot_inf(inf, assumedgrad, maxtime=3*maxtime), filename * ".pdf")
    end
    
    println("")
    return df

end


function _mnist_data(folder::ASCIIString,
                     X_train::Union{Void,Array{Float64,2}}=nothing, y_train::Union{Void,Array{Float64,1}}=nothing,
                     X_test::Union{Void,Array{Float64,2}}=nothing,  y_test::Union{Void,Array{Float64,1}}=nothing)

    if X_train == nothing || y_train == nothing || X_test == nothing || y_test == nothing
        println("Reading data from ./data/$folder...")
    else
        println("Data already present - nothing to read.")
    end

    if X_train == nothing
        X_train = readdlm("./data/$folder/X_train.dlm")
    end
    if y_train == nothing
        y_train = vec(readdlm("./data/$folder/y_train.dlm"))
    end
    if X_test == nothing
        X_test  = readdlm("./data/$folder/X_test.dlm")
    end
    if y_test == nothing
        y_test  = vec(readdlm("./data/$folder/y_test.dlm"))
    end

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
    time = inf[end, :time]
    iterrate = inf[end, :iter] / inf[end, :time]

    LsoBase.push_truth!(inf, obj.f, obj.g)  # true f and opt values

    # acc
    acc_train = LsoBase.acc(y_train, Obj.logreg_predict(w, X_train))
    acc_test  = LsoBase.acc(y_test, Obj.logreg_predict(w, X_test))
    println("\n  Training set acc: $acc_train")
    println(  "      Test set acc: $acc_test")
    println(  "  Iterations / sec: $iterrate\n")

    return inf, obj, w, acc_train, acc_test, time, iterrate

end