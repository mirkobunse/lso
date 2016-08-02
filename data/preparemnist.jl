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

function preparemnist(p_X, p_y, Xy)
    X, y = Xy

    println("Selecting subset...")
    select = convert(BitArray{1}, [ yi == 1.0 || yi == 7.0 for yi in y ])
    X = X[:, select]'
    y = y[select]

    println("Shuffling...")
    perm = randperm(length(y))
    X = X[perm, :]
    y = y[perm]

    println("Storing data into $p_X and $p_y...")
    writedlm(p_X, X)
    writedlm(p_y, y)
end

println("\nPreparing MNIST training data...")
preparemnist("./X_train.dlm", "./y_train.dlm", MNIST.traindata())
gc()    # cleanup

println("\nPreparing MNIST test data...")
preparemnist("./X_test.dlm", "./y_test.dlm", MNIST.testdata())
gc()    # cleanup

println("\nDone.\n")
return nothing