module Eval

import LsoBase

# include ./eval/*
# LsoBase.includedir(dirname(@__FILE__)*"/eval")

include(dirname(@__FILE__)*"/eval/rosenbrock.jl")
include(dirname(@__FILE__)*"/eval/rand.jl")
include(dirname(@__FILE__)*"/eval/mnist.jl")

end