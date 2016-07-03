module LsoBase


using DataFrames


"""
    newinf()

    Create new DataFrame to hold iteration info. Will have columns w, opt, iter and lsiter.
"""
new_inf() = DataFrame(
                w = Array{Float64, 1}[],
                f = Float64[],
                opt = Float64[],
                iter = Int32[],
                lsiter = Int32[]
            )


"""
    pushinf!(inf, w, opt, iter, lsiter)

    Will push the given iteration info to DataFrame inf.
"""
function push_inf!(inf::DataFrame, w::Array{Float64, 1},
                   f::Float64, opt::Float64, iter::Int32, lsiter::Int32)
    push!(inf, (w, f, opt, iter, lsiter))
end


end