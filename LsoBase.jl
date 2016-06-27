module LsoBase


using DataFrames
using Gadfly
using Colors


"""
    newinf()

    Create new DataFrame to hold iteration info. Will have columns w, opt, iter and lsiter.
"""
newinf() = DataFrame(
               w = Array{Float64, 1}[],
               opt = Float64[],
               iter = Int32[],
               lsiter = Int32[]
           )


"""
    pushinf!(inf, w, opt, iter, lsiter)

    Will push the given iteration info to DataFrame inf.
"""
pushinf!(inf::DataFrame, w::Array{Float64, 1}, opt::Float64, iter::Int32, lsiter::Int32) = push!(inf, (
                                                                                               w,
                                                                                               opt,
                                                                                               iter,
                                                                                               lsiter
                                                                                           ))


"""
    plotopt(inf)

    Plot development of optimality, as given by the DataFrame inf.
"""
plotopt(inf::DataFrame) = plot(inf, x=:iter, y=:opt, Scale.y_log10, Scale.x_continuous(format=:plain),
                               Geom.line, Guide.xlabel("Iteration"), Guide.ylabel("Optimality ‖∇f(x)‖∞"))


end