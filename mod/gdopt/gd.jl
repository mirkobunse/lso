import Obj
import Obj.Objective

gd() = GdOptimizer(

    # name
    "gd",

    # initial state
    EmptyState(),

    # update
    function (obj::Objective, k::Int32, w::Array{Float64,1}, b::Array{Int32,1}, inf, state::State)
        gw = Obj.g(obj, w)
        return Obj.f(obj, w), gw, -gw, state
    end
    
)