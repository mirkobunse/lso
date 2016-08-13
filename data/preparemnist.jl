################
# Script for getting MNIST data, selecting ones and sevens, shuffling
# and storing that subset in the data folder.
# 
# Run this script standalone (instead of using your dev kernel and without
# using the .juliarc.jl of this project), like
# cd data
# julia preparemnist.jl
################

# Pkg.add("MNIST")
import MNIST

function preparemnist(Xy, func1, func2, p_X, p_y, maxclasssize)
    X, y = Xy

    # split into classes
    X_a = X[:,func1(y)]    # X[:,(y.==7.0)]
    X_b = X[:,func2(y)]    # X[:,(y.!=7.0)]

    # cleanup
    X = nothing
    y = nothing
    gc()

    classsize = min(size(X_a)[2], size(X_b)[2], maxclasssize)
    println("Stratifying data to have 2 * $classsize examples (dimension $(size(X_a)[1]))...")

    # subsample
    X_a = X_a[:, randperm(size(X_a)[2])[1:classsize]]
    X_b = X_b[:, randperm(size(X_b)[2])[1:classsize]]

    # normalize and transpose for convenience
    X_a = X_a' ./ 255.0
    X_b = X_b' ./ 255.0

    # bring together
    y = vcat(repeat([1.0], inner=[classsize]), repeat([-1.0], inner=[classsize]))
    X = vcat(X_a, X_b)
    X_a = nothing
    X_b = nothing
    gc()

    println("Shuffling...")
    perm = randperm(2*classsize)
    X = X[perm,:]
    y = y[perm]

    println("Storing data into $p_X and $p_y...")
    writedlm(p_X, X)
    writedlm(p_y, y)
end


################
# 7-vs-all split
################
mkdir("./seven_vs_all")
println("\nPreparing 7-vs-all test data...")
preparemnist(MNIST.traindata(), (y -> y .== 7.0), (y -> y .!= 7.0),
             "./seven_vs_all/X_test.dlm", "./seven_vs_all/y_test.dlm", 2500)
gc()

println("\nPreparing 7-vs-all training data...")
preparemnist(MNIST.testdata(), (y -> y .== 7.0), (y -> y .!= 7.0),
             "./seven_vs_all/X_train.dlm", "./seven_vs_all/y_train.dlm", 1000)


################
# 7-vs-6 split
################
mkdir("./seven_vs_six")
println("\nPreparing 7-vs-six test data...")
preparemnist(MNIST.traindata(), (y -> y .== 7.0), (y -> y .== 6.0),
             "./seven_vs_six/X_test.dlm", "./seven_vs_six/y_test.dlm", 2500)
gc()

println("\nPreparing 7-vs-all training data...")
preparemnist(MNIST.testdata(), (y -> y .== 7.0), (y -> y .== 6.0),
             "./seven_vs_six/X_train.dlm", "./seven_vs_six/y_train.dlm", 1000)


println("\nDone.\n")


return nothing