# set paths
push!(LOAD_PATH, pwd()*"/mod")
push!(LOAD_PATH, pwd()*"/tst")

# import modules
import Opt
import Obj

# import test modules and run tests
import OptTst.tst
import ObjTst.tst
println("")
OptTst.tst()
ObjTst.tst()

# reset rng seed
srand(1337)


return true