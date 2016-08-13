module Tst


using Base.Test

import LsoBase


# include ./tst/*
LsoBase.includedir(dirname(@__FILE__)*"/tst")


"""
    tst()

    Run all tests.
"""
function all()
    for exp in names(Tst, true) # iterate over exported members
        if (exp != :Tst && exp != :__META__ && exp != :eval && exp != :all && !startswith(string(exp), "_"))
            eval( :(($exp)()) )  # call exp as function
        end
    end
end


end