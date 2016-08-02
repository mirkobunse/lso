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

function preparemnist(Xy, p_X, p_y, maxclasssize)
    X, y = Xy
    X = X' # transpose for convenience

    # recode labels
    X_a = X[(y.==1.0),:]
    X_b = X[(y.==2.0),:]
    classsize = min(size(X_a)[1], size(X_b)[1], maxclasssize)

    println("Stratifying data to have 2 * $classsize examples...")
    X_a = X_a[randperm(size(X_a)[1])[1:classsize], :]
    X_b = X_b[randperm(size(X_b)[1])[1:classsize], :]

    # bring together
    y = vcat(repeat([1.0], inner=[classsize]), repeat([-1.0], inner=[classsize]))
    X = vcat(X_a, X_b)

    println("Shuffling...")
    perm = randperm(2*classsize)
    X = X[perm,:]
    y = y[perm]

    println("Storing data into $p_X and $p_y...")
    writedlm(p_X, X)
    writedlm(p_y, y)
end


println("\nPreparing MNIST test data...")
preparemnist(MNIST.traindata(), "./X_test.dlm", "./y_test.dlm", 2500)

println("\nPreparing MNIST training data...")
preparemnist(MNIST.testdata(),"./X_train.dlm", "./y_train.dlm", 500)

println("\nDone.\n")
return nothing