#=
A simple script for cleaning up temporary files.
=#

code_dir = dirname(@__DIR__)

for (root, dirs, files) in Base.Filesystem.walkdir(code_dir)
    for f in files
        if endswith(f, ".DS_Store")
            rm(joinpath(root, f); force = true)
        end
        # https://github.com/JuliaLang/Pkg.jl/issues/3014
        if basename(f) == "LocalPreferences.toml"
            rm(joinpath(root, f); force = true)
        end
    end
end
