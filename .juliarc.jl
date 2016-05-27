# imports all modules from a given directory
function importdir(dir::AbstractString)
	push!(LOAD_PATH, pwd()*"/"*dir)
	for filename in readdir(dir)
		if endswith(filename, ".jl")
			:(@everywhere import replace(filename, ".jl", "")) # filename without ".jl"
			println("Imported $(replace(filename, ".jl", "")) everywhere.")
		end
	end
end

# includes all julia files from a given directory
function includedir(dir::AbstractString)
    for filename in readdir(dir)
        if endswith(filename, ".jl")
            include(dir*"/"*filename) # concat
            println("Included $(dir*"/"*filename).")
        end
    end
end

# import modules
importdir("mod")

# perform tests
println("\nTesting...")
srand(1337)
includedir("tst")

# reset rng seed
srand(1337)

println("Done.")