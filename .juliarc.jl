MODULE_NAMES = ["LsoBase", "Obj", "Ls", "GdOpt", "Plotting", "Tst", "Eval"]


# import modules
println("Importing modules...")
push!(LOAD_PATH, pwd()*"/mod")
for name in MODULE_NAMES
    try
        eval(Expr(:import, symbol(name))) # metaprogramming: import <name>
        println("Imported $(name)")
    catch e
        println("ERROR: Could not import $(name):\n$e")
    end
end

# ease reloading
reload(name::AbstractString) = Base.reload(name)
function reload()
    for name in MODULE_NAMES
        reload(name)
    end
end

# run tests
#Tst.all()