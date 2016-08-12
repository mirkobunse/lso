module Plotting


using DataFrames
using Gadfly
using Colors

import LsoBase
import Obj


"""
    display_plot(plot)

    Display plot the default way (e.g., in browser).
"""
function display_plot(plot)
    Base.display(plot)
    return nothing
end


"""
    draw_plot(plot [, filename])

    Draw plot to file.
"""
function draw_plot(plot, filename::ASCIIString="out.pdf")
    draw(PDF(filename, 15cm, 9cm), plot)
end


"""
    plot_inf(inf)

    Plot development of optimality and function value, as given by the DataFrame inf.
    Returns Gadfly plot object.
"""
function plot_inf(inf::DataFrame, obj=nothing, assumedgrad=true; vlines=nothing)

    f    = inf[:f]
    opt  = inf[:opt]
    if obj != nothing
      println("Computing true f and opt values...")
        f = [ obj.f(w)               for w in inf[:w] ]
      opt = [ vecnorm(obj.g(w), Inf) for w in inf[:w] ]
    end

    println("Plotting...")
    df = vcat(
        DataFrame(
            x = inf[:time],
            y = opt,
            name = [ "Optimality" for i=1:length(inf[:time]) ]  # ‖∇f(x)‖∞
        ),
        DataFrame(
            x = inf[:time],
            y = f,
            name = [ "Function Value" for i=1:length(inf[:time]) ]  # f(x)
        )
    )
    colorscale = Scale.color_discrete_manual(LCHab{Float64}(65.0,70.0,0.0),
                                             LCHab{Float64}(70.0,60.0,240.0),
                                             order=[2,1])
    if assumedgrad
      df = vcat(df,
          DataFrame(
              x = inf[:time],
              y = inf[:opt],
              name = [ "Assumed Opt." for i=1:length(inf[:time]) ]  # ‖∇f(x)‖∞
          )
      )
      colorscale = Scale.color_discrete_manual(LCHab{Float64}(65.0,70.0,0.0),
                                               LCHab{Float64}(70.0,60.0,240.0),
                                               LCHab{Float64}(93.7,125.2,100.43478260869566),
                                               order=[2,1,3])
    end

    args = [Geom.line, Scale.y_log10(minvalue=1e-6, maxvalue=1e6), Scale.x_continuous(labels = (x -> @sprintf "%6ds" x)),
            # labels = function (x)
            #     label::ASCIIString = @sprintf "%6ds" x
            #     iters = inf[ inf[:time] .>= x, :iter ]
            #     if length(iters) > 0
            #         label *= @sprintf "\n(%d)" iters[1]
            #     # else
            #     #     iters = inf[ inf[:time] .<= x, :iter ]
            #     #     if length(iters) > 0
            #     #         label *= @sprintf "\n(%6d:\n%6.3fs)" inf[end, :time] iters[end]
            #     #     end
            #     end
            #     return label
            # end),
            Guide.xticks(orientation=:horizontal), Guide.xlabel("\n\n   Time  \n(Iteration)"), Guide.ylabel(""),
            Guide.colorkey(""), colorscale]

    if vlines == nothing
      return Gadfly.plot(df, x=:x, y=:y, color=:name, args...)
    else
      return Gadfly.plot(df, x=:x, y=:y, color=:name, xintercept=vlines, Geom.vline, args...)
    end

end


end