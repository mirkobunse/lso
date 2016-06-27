
reload(name::AbstractString) = Base.reload(name)
function reload()
    reloaddir(pwd(), true)
end

# import modules
println("Importing modules...")
importdir(pwd(), true)

# run tests
#println("")
#OptTst.tst()
#ObjTst.tst()
