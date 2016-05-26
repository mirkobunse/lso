# includes all julia files from a given directory
function includedir(dir::AbstractString)
	for filename in readdir(dir)
		if endswith(filename, ".jl")
			include(dir*"/"*filename) # concat
			println("Included ", dir*"/"*filename)
		end
	end
end

# imports all modules from a given directory
function importdir(dir::AbstractString)
	push!(LOAD_PATH, pwd()*"/"*dir)
	for filename in readdir(dir)
		if endswith(filename, ".jl")
			:(import replace(filename, ".jl", "")) # filename without ".jl"
			println("Imported ", replace(filename, ".jl", ""))
		end
	end
end

importdir("mod")
includedir("tst")