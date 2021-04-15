include("data.jl")
include("grid.jl")
include("quadrature.jl")

if "CUDA" in ARGS
    include("cuda.jl")
    include("gpu/data.jl")
end
