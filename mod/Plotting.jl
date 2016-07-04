module Plotting


using DataFrames
using Gadfly
using Colors

import LsoBase


"""
    display_inf(inf)

    Like plot_inf(inf), but instead of returning plot, display it.
"""
function display_inf(inf::DataFrame)
    Base.display(plot_inf(inf))
    return nothing
end


"""
    plot_inf(inf)

    Plot development of optimality and function value, as given by the DataFrame inf.
"""
function plot_inf(inf::DataFrame)
    println("Plotting...")
    df = vcat(
        DataFrame(
            x = inf[:iter],
            y = inf[:opt],
            name = [ "Optimality" for i=1:length(inf[:iter]) ]  # ‖∇f(x)‖∞
        ),
        DataFrame(
            x = inf[:iter],
            y = inf[:f],
            name = [ "Function Value" for i=1:length(inf[:iter]) ]  # f(x)
        )
    )
    return Gadfly.plot(df, x=:x, y=:y, color=:name, Geom.line,
                       Scale.y_log10, Scale.x_continuous(format=:plain),
                       Guide.xlabel("Iteration"), Guide.ylabel(""), Guide.colorkey(""))
end


end