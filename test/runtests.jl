include("data.jl")
include("grid.jl")

if "CUDA" in ARGS
    include("cuda.jl")
    include("gpu/data.jl")
end
