# 
# Use this in an active julia kernel:
# 
# include("prepareresults.jl")
# df = prepareresults()
# 
# This way, a nicely aggregated DataFrame is returned.
# 
using DataFrames


function prepareresults(folder::ASCIIString="./seven_vs_all")

    # concatenate DataFrames
    df = DataFrame()
    for seed in readdir(folder)
        if isdir(folder*"/"*seed)
            accfile = folder*"/"*seed*"/acc.csv"
            if isfile(accfile)
                println("Reading $accfile...")
                df = vcat(df, readtable(accfile))
            end
        end
    end

    # only consider LogReg and Boosting (ensemble vote)
    df = df[convert(Array{Bool,1}, [ c == "LogReg" || c == "Boosting B1-3" for c in df[:classifier]]),:]

    # rename columns so it fits on screen
    rename!(df, [:batchsize, :classifier, :trainsize, :trainacc, :testacc, :iterrate],
                [:batch,     :learner,    :numex,     :train,    :test,    :rate])

    # shorten classifier name
    df[:learner] = [ (learner == "LogReg") ? "LogReg" : "Boosting" for learner in df[:learner] ]

    # print some info
    meanex = mean(df[df[:learner] .== "Boosting",:][:numex])
    stdex  = stdm(df[df[:learner] .== "Boosting",:][:numex], meanex)
    println("\nBoosting used $meanex examples on average (std: $stdex)\n")

    # compute std and mean
    df = aggregate(df, [:gdopt, :ls, :batch, :data, :learner], [mean, std])

    # select and reorder columns
    df = df[:, [:learner, :gdopt, :batch,
                :test_mean, :test_std, :time_mean, :time_std,
                :rate_mean, :rate_std, :train_mean, :train_std ]]

    # sort
    sort!(df, rev=[true, false, false, false], cols=[:test_mean, :test_std, :time_mean, :time_std])

    return df

end