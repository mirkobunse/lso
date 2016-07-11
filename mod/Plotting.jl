module Plotting


using DataFrames
using Gadfly
using Colors

import LsoBase


"""
    display_inf(inf)

    Like plot_inf(inf), but instead of returning the plot, display it.
"""
function display_inf(inf::DataFrame)
    Base.display(plot_inf(inf))
    return nothing
end


"""
    draw_inf(inf [, filename])

    Like plot_inf(inf), but instead of returning the plot, draw it to a file.
"""
function draw_inf(inf::DataFrame, filename::ASCIIString="out.pdf")
    draw(PDF(filename, 15cm, 9cm), plot_inf(inf))
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
#    return inf[ inf[:iter] .== 4 ,[:iter, :time]]
    return Gadfly.plot(df, x=:x, y=:y, color=:name, Geom.line,
                       Scale.y_log10(minvalue=1e-6, maxvalue=1e6), Scale.x_continuous(
                       labels = function (x)
                           label::ASCIIString = @sprintf "%6d" x
                           times = inf[ inf[:iter] .== x, :time ]
                           if length(times) > 0
                               label *= @sprintf "\n%6.3fs" times[end]
                           else
                               times = inf[ inf[:iter] .<= x, :time ]
                               if length(times) > 0
                                   label *= @sprintf "\n(%6d:\n%6.3fs)" inf[end, :iter] times[end]
                               end
                           end
                           return label
                       end),
                       Guide.xticks(orientation=:horizontal),
                       Guide.xlabel("\n\nIteration\n (Time)"), Guide.ylabel(""), Guide.colorkey(""))
end


end