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
    newinf()

    Create new DataFrame to hold iteration info. Will have columns w, opt, iter and lsiter.
"""
new_inf() = DataFrame(
                w = Array{Float64, 1}[],
                f = Float64[],
                opt = Float64[],
                iter = Int32[],
                lsiter = Int32[],
                time = Float64[]
            )


"""
    pushinf!(inf, w, opt, iter, lsiter)

    Will push the given iteration info to DataFrame inf.
"""
function push_inf!(inf::DataFrame, w::Array{Float64, 1},
                   f::Float64, opt::Float64, iter::Int32, lsiter::Int32, time::Float64=0.0)
    push!(inf, (w, f, opt, iter, lsiter, time))
end


acc(y_true::Array{Float64, 1}, y_pred::Array{Float64, 1}) = mean(y_true .== y_pred)


end