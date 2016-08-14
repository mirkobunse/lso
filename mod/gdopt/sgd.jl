import Obj
import Obj.Objective

sgd() = GdOptimizer(

    # name
    "sgd",

    # initial state
    EmptyState(),

    # update
    function (obj::Objective, k::Int, w::Array{Float64,1}, b::Array{Int,1}, inf, state::State)
        gw = Obj.g(obj, w, b)
        return Obj.f(obj, w, b), gw, -gw, state
    end
    
)
