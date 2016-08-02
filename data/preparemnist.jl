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

function preparemnist(Xy, p_X, p_y, maxnumones)
    X, y = Xy
    X = X' # transpose for convenience

    # recode labels: "1.0" vs all
    X_one = X[(y.==1.0),:]
    X_all = X[(y.!=1.0),:]
    numones = min(size(X_one)[1], maxnumones)

    println("Stratifying data to have 2 * $numones examples...")
    X_all = X_all[randperm(size(X_all)[1])[1:numones], :]

    # bring together
    y = vcat(repeat([1.0], inner=[numones]), repeat([-1.0], inner=[numones]))
    X = vcat(X_one, X_all)

    println("Shuffling...")
    perm = randperm(2*numones)
    X = X[perm,:]
    y = y[perm]

    println("Storing data into $p_X and $p_y...")
    writedlm(p_X, X)
    writedlm(p_y, y)
end

println("\nPreparing MNIST training data...")
preparemnist(MNIST.testdata(),"./X_train.dlm", "./y_train.dlm", 1000)

println("\nPreparing MNIST test data...")
preparemnist(MNIST.traindata(), "./X_test.dlm", "./y_test.dlm", 5000)

println("\nDone.\n")
return nothing