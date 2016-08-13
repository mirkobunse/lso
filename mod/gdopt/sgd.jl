import Obj
import Obj.Objective

sgd() = GdOptimizer(

    # initial state
    EmptyState(),

    # update
    function (obj::Objective, k::Int32, w::Array{Float64,1}, b::Array{Int32,1}, inf, state::State)
        gw = Obj.g(obj, w, b)
        return Obj.f(obj, w, b), gw, -gw, state
    end
    
)