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
    for exported in names(Tst, true)
        if (exported != :Tst && exported != :__META__ && exported != :eval && exported != :all)
            eval( :(($exported)()) )  # call exported as function
        end
    end
end


end