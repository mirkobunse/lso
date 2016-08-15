import Obj
import Obj.Objective

type SvrgState <: State
    w_est::Array{Float64,1}     # w estimate
    gw_est::Array{Float64,1}    # μ (= full gradient of w estimate)
end

"""
    svrg([estiter, strategy])

    The estiter parameter describes the number of iterations before a new w estimate is
    obtained. The strategy to obtain a new w estimate is set by the strategy parameter.
    The following stategies exist:
    - :last     Snapshat of very last iteration (default)
    - :rand     Random snapshot of last iterations
    - :avg      Average of last iterations
"""
svrg(estiter::Int32=10, strategy::Symbol=:last) = GdOptimizer(

    # name
    "svrg-"*string(estiter)*"-"*string(strategy),

    # initial state
    SvrgState(Float64[], Float64[]),    # correct values will be set in first iteration

    # update
    function (obj::Objective, k::Int32, w::Array{Float64,1}, b::Array{Int32,1}, inf, state::State)
        return _svrg_update(obj, k, w, b, inf, state, estiter, strategy)
    end

)

@fastmath function _svrg_update(obj::Objective, k::Int32, w::Array{Float64,1}, b::Array{Int32,1}, inf, state::State,
                                estiter::Int32, strategy::Symbol)

    gw = Obj.g(obj, w, b)

    # svrg estimation update
    if (k-1) % estiter == 0
        if strategy == :last
            state.w_est  = w
        elseif strategy == :rand
            try 
                state.w_est = inf[:w][ end-rand(1:estiter)+1 ]
            catch e
                state.w_est = w
            end
        elseif strategy == :avg
            try
                state.w_est = mean(inf[:w][ (end-estiter+1):end ], 2)
            catch e
                state.w_est = w
           end
        end
        state.gw_est = Obj.g(obj, state.w_est)
    end
    gw_est_b = Obj.g(obj, state.w_est, b)   # stochastic gradient of w estimate

   s = -gw + gw_est_b - state.gw_est # -g(w) + g(w_est) - μ

    return Obj.f(obj, w, b), gw, s, state

end