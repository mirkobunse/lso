# includes all julia files from a given directory
function includefiles(dir::AbstractString)
	for filename in readdir(dir)
		if endswith(filename, ".jl")
			include(dir*"/"*filename) # concat
			println("Included ", dir*"/"*filename)
		end
	end
end

includefiles("src")
includefiles("tst")