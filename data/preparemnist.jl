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

function preparemnist(Xy, func1, func2, p_X, p_y, maxclasssize, seed)
    X, y = Xy

    srand(seed)

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
# folders
################
for folder in ["./seven_vs_all", "../results", "../results/seven_vs_all",
               "./seven_vs_six", "../results/seven_vs_six"]
    try
        mkdir(folder)
    end
end

################
# splits
################
for seed in 1337:(1337+9)
    println("\n", repeat("-", 10), "Seed $seed", repeat("-", 10))
    try
        mkdir("./seven_vs_all/$seed")
    end
    try
        mkdir("./seven_vs_six/$seed")
    end
    try
        mkdir("../results/seven_vs_all/$seed")
    end
    try
        mkdir("../results/seven_vs_six/$seed")
    end

    println("\nPreparing 7-vs-all TEST data...")
    preparemnist(MNIST.traindata(), (y -> y .== 7.0), (y -> y .!= 7.0),
                 "./seven_vs_all/$seed/X_test.dlm", "./seven_vs_all/$seed/y_test.dlm", 2500, seed)
    gc()

    println("\nPreparing 7-vs-all TRAINING data...")
    preparemnist(MNIST.testdata(), (y -> y .== 7.0), (y -> y .!= 7.0),
                 "./seven_vs_all/$seed/X_train.dlm", "./seven_vs_all/$seed/y_train.dlm", 1000, seed)
    gc()

    println("\nPreparing 7-vs-six TEST data...")
    preparemnist(MNIST.traindata(), (y -> y .== 7.0), (y -> y .== 6.0),
                 "./seven_vs_six/$seed/X_test.dlm", "./seven_vs_six/$seed/y_test.dlm", 2500, seed)
    gc()

    println("\nPreparing 7-vs-six TRAINING data...")
    preparemnist(MNIST.testdata(), (y -> y .== 7.0), (y -> y .== 6.0),
                 "./seven_vs_six/$seed/X_train.dlm", "./seven_vs_six/$seed/y_train.dlm", 1000, seed)
    gc()
end

println("\nDone.\n")

return nothing