
# import all modules from a given directory relative to working directory
function import_localdir(dir::AbstractString)
   push!(LOAD_PATH, pwd()*"/"*dir)
   for filename in readdir(dir)
       if endswith(filename, ".jl")
            eval(Expr(:import, symbol(replace(filename, ".jl", "")))) # metaprogramming: import <filename>
            println("Imported $(replace(filename, ".jl", "")).")
       end
    end
end

# import modules
@everywhere import_localdir("mod")
import_localdir("tst")

# run tests
println("")
OptTst.tst()
ObjTst.tst()

# reset rng seed
srand(1337)


return true