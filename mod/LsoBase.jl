module LsoBase


using DataFrames


"""
    includedir(dir)

    Include all non-hidden .jl files from the given directory
"""
function includedir(dir::AbstractString)
    for filename in readdir(dir)
        if isfile(dir*"/"*filename) && endswith(filename, ".jl") && !startswith(filename, ".")
            include(dir*"/"*filename)
        end
    end
end


"""
    new_inf()

    Create new DataFrame to hold iteration info.
    Will have columns f, w, opt, iter, lsiter and time.
"""
new_inf() = DataFrame(
                w      = Array{Float64, 1}[],
                f      = Float64[],
                opt    = Float64[],
                iter   = Int32[],
                lsiter = Int32[],
                time   = Float64[]
            )

"""
    new_acc()

    Create new DataFrame to hold classifier info.
    Will have columns gdopt, ls, classifier data, trainsize,
    trainacc, testacc, time and iterrate.
"""
new_acc() = DataFrame(
                gdopt      = ASCIIString[],
                ls         = ASCIIString[],
                data       = ASCIIString[],
                classifier = ASCIIString[],
                trainsize  = Int32[],
                trainacc   = Float64[],
                testacc    = Float64[],
                time       = Float64[],
                iterrate   = Float64[]
            )


"""
    push_inf!(inf, w, opt, iter, lsiter)

    Will push the given iteration info to DataFrame inf.
"""
function push_inf!(inf::DataFrame, w::Array{Float64, 1},
                   f::Float64, opt::Float64, iter::Int32, lsiter::Int32, time::Float64=0.0)
    push!(inf, (w, f, opt, iter, lsiter, time))
end


"""
    push_acc!(acc, gdopt, ls, data, trainsize, trainacc, testacc, time, iterrate)

    Will push the given classifier info to DataFrame acc.
"""
function push_acc!(acc::DataFrame, gdopt::ASCIIString, ls::ASCIIString, data::ASCIIString,
                   classifier::ASCIIString, trainsize::Int32, trainacc::Float64, testacc::Float64,
                   time::Float64, iterrate::Float64)
    push!(acc, (gdopt, ls, data, classifier, trainsize, trainacc, testacc, time, iterrate))
end


acc(y_true::Array{Float64, 1}, y_pred::Array{Float64, 1}) = mean(y_true .== y_pred)


end