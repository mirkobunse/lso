module LsoBase


using Gadfly
using Colors


type IterInfo
    w::Array{Float64, 1}
    opt::Float64
    iter::Integer
    lsiter::Integer
end


"""
    plotinf(infarr [, ylabel])

    Plot development of optimality in IterInfo array.
"""
function plotinf(infarr, ylabel="Optimality ‖∇f(x)‖∞")
    opts = [info.opt for info in infarr]
    plot(x=1:length(opts), y=opts, Scale.y_log10, Scale.x_continuous, Geom.line, Guide.xlabel("Iteration"), Guide.ylabel(ylabel))
end


end