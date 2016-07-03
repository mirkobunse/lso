
reload(name::AbstractString) = Base.reload(name)
function reload()
    for dir in ["opt", "obj", "eval"]
        reloaddir(pwd()*"/"*dir, false)
    end
end

# import modules
println("Importing modules...")
push!(LOAD_PATH, pwd())
import LsoBase
for dir in ["opt", "obj", "eval"]
    importdir(pwd()*"/"*dir, false)
end

# run tests
#println("")
#OptTst.tst()
#ObjTst.tst()
