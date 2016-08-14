using Colors

import Ls
import Ls.LineSearch
import GdOpt
import GdOpt.GdOptimizer
import Obj
import Plotting


""" Quick test function for gradient descent. """
mnist_gd() = mnist(GdOpt.gd(), Ls.bt())

""" Quick test function for stochastic gradient descent. """
mnist_sgd(batchsize=1) = mnist(GdOpt.sgd(), Ls.sbt(), ϵ=0.0, batchsize=batchsize)

""" Quick test function for SVRG. """
function mnist_svrg(; batchsize=10, estiter=10, strategy=:last)
    mnist(GdOpt.svrg(estiter, strategy), Ls.sbt(), batchsize=batchsize, assumedgrad=true)
end


"""
    mnist(gdopt, ls, [, folder, seed; ϵ, maxtime, batchsize, assumedgrad])

    Evaluate optimizer on MNIST data.
"""
function mnist(gdopt::GdOptimizer, ls::LineSearch, folder::ASCIIString="seven_vs_all", seed::Int32=1337;
               ϵ::Float64=1e-3, maxtime::Float64=60.0, batchsize::Int32=-1, assumedgrad=false)

    srand(seed)

    println("Reading data from ./data/$folder...")
    X_train = readdlm("./data/$folder/X_train.dlm")
    y_train = vec(readdlm("./data/$folder/y_train.dlm"))
    X_test  = readdlm("./data/$folder/X_test.dlm")
    y_test  = vec(readdlm("./data/$folder/y_test.dlm"))

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